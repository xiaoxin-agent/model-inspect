"""Bounded, streaming metadata reader for legacy ``lmgg`` GGML v1 files.

This format has no vocabulary scores and no tensor alignment padding. The
reader supports the F32 and F16 tensor types present in the initial fixture;
it seeks over payloads rather than loading model weights into memory.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import struct
from typing import BinaryIO


MAGIC = b"lmgg"
MAX_VOCAB = 1_000_000
MAX_TOKEN_BYTES = 4095
MAX_TENSOR_NAME_BYTES = 4095
MAX_DIMS = 4
MAX_TENSORS = 1_000_000
UINT64_MAX = (1 << 64) - 1

# Legacy ggml_type IDs used by the fixture. Other IDs fail closed so the
# parser never guesses a quantized tensor's payload size.
TENSOR_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
}

FTYPES = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
}


class GGMLFormatError(ValueError):
    """The file is unsupported or its metadata is malformed/truncated."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    dtype_id: int
    elements: int
    data_bytes: int
    data_offset: int


@dataclass(frozen=True)
class Inspection:
    path: str
    format: str
    format_version: int
    file_size: int
    hyperparameters: dict[str, int]
    ftype: str
    vocabulary_count: int
    tensor_count: int
    tensor_elements: int
    tensor_data_bytes: int
    dtype_counts: dict[str, int]
    tensors: list[TensorInfo]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        return result


def _read_exact(stream: BinaryIO, size: int, context: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise GGMLFormatError(f"truncated file while reading {context}")
    return data


def _skip(stream: BinaryIO, size: int, file_size: int, context: str) -> None:
    if size < 0:
        raise GGMLFormatError(f"invalid negative size while skipping {context}")
    target = stream.tell() + size
    if target > file_size:
        raise GGMLFormatError(f"{context} extends past end of file")
    stream.seek(target)


def inspect(path: str | os.PathLike[str]) -> Inspection:
    """Inspect an ``lmgg`` GGML v1 file without loading tensor data."""
    model_path = Path(path)
    try:
        file_size = model_path.stat().st_size
        stream = model_path.open("rb")
    except OSError:
        raise

    with stream:
        magic = _read_exact(stream, 4, "magic")
        if magic != MAGIC:
            if magic == b"GGUF":
                raise GGMLFormatError("this is GGUF, not legacy GGML v1 (lmgg)")
            raise GGMLFormatError(f"unsupported model magic {magic!r}; expected b'lmgg'")

        header = struct.unpack("<7I", _read_exact(stream, 28, "GGML v1 header"))
        n_vocab, n_embd, n_mult, n_head, n_layer, n_rot, ftype_id = header
        if n_vocab == 0 or n_vocab > MAX_VOCAB:
            raise GGMLFormatError(f"unreasonable vocabulary count: {n_vocab}")
        if n_embd == 0 or n_head == 0 or n_layer == 0:
            raise GGMLFormatError("invalid zero-valued core hyperparameter")

        # GGML v1 stores each vocabulary item as uint32 length + bytes. It
        # predates the score field used by GGMF/GGJT.
        for token_id in range(n_vocab):
            token_len = struct.unpack("<I", _read_exact(stream, 4, f"token {token_id} length"))[0]
            if token_len > MAX_TOKEN_BYTES:
                raise GGMLFormatError(f"token {token_id} length {token_len} exceeds the v1 limit")
            _skip(stream, token_len, file_size, f"token {token_id}")

        tensors: list[TensorInfo] = []
        names: set[str] = set()
        dtype_counts: Counter[str] = Counter()
        total_elements = 0
        total_data_bytes = 0

        while stream.tell() < file_size:
            if len(tensors) >= MAX_TENSORS:
                raise GGMLFormatError(f"tensor count exceeds safety limit ({MAX_TENSORS})")

            tensor_start = stream.tell()
            ndims, name_len, dtype_id = struct.unpack(
                "<3I", _read_exact(stream, 12, f"tensor header at offset {tensor_start}")
            )
            if not 1 <= ndims <= MAX_DIMS:
                raise GGMLFormatError(f"tensor at offset {tensor_start} has invalid dimension count {ndims}")
            if not 1 <= name_len <= MAX_TENSOR_NAME_BYTES:
                raise GGMLFormatError(f"tensor at offset {tensor_start} has invalid name length {name_len}")

            dims_data = _read_exact(stream, ndims * 4, f"tensor dimensions at offset {tensor_start}")
            shape = list(struct.unpack("<" + "I" * ndims, dims_data))
            if any(dim == 0 for dim in shape):
                raise GGMLFormatError(f"tensor at offset {tensor_start} has a zero-sized dimension")

            raw_name = _read_exact(stream, name_len, f"tensor name at offset {tensor_start}")
            try:
                name = raw_name.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise GGMLFormatError(f"tensor at offset {tensor_start} has a non-UTF-8 name") from exc
            if name in names:
                raise GGMLFormatError(f"duplicate tensor name {name!r}")
            names.add(name)

            type_info = TENSOR_TYPES.get(dtype_id)
            if type_info is None:
                raise GGMLFormatError(
                    f"tensor {name!r} uses unsupported legacy dtype id {dtype_id}; "
                    "this initial reader supports F32 (0) and F16 (1)"
                )
            dtype_name, block_elements, block_bytes = type_info

            elements = 1
            for dim in shape:
                elements *= dim
                if elements > UINT64_MAX:
                    raise GGMLFormatError(f"tensor {name!r} element count overflows uint64")
            if elements % block_elements:
                raise GGMLFormatError(f"tensor {name!r} element count is not a multiple of its block size")
            data_bytes = elements // block_elements * block_bytes
            data_offset = stream.tell()
            _skip(stream, data_bytes, file_size, f"tensor {name!r} payload")

            tensors.append(
                TensorInfo(
                    name=name,
                    shape=shape,
                    dtype=dtype_name,
                    dtype_id=dtype_id,
                    elements=elements,
                    data_bytes=data_bytes,
                    data_offset=data_offset,
                )
            )
            dtype_counts[dtype_name] += 1
            total_elements += elements
            total_data_bytes += data_bytes
            if total_elements > UINT64_MAX or total_data_bytes > UINT64_MAX:
                raise GGMLFormatError("aggregate tensor size overflows uint64")

        if not tensors:
            raise GGMLFormatError("file contains no tensors")

    hyperparameters = {
        "n_vocab": n_vocab,
        "n_embd": n_embd,
        "n_mult": n_mult,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_rot": n_rot,
    }
    return Inspection(
        path=str(model_path),
        format="GGML",
        format_version=1,
        file_size=file_size,
        hyperparameters=hyperparameters,
        ftype=FTYPES.get(ftype_id, f"UNKNOWN_{ftype_id}"),
        vocabulary_count=n_vocab,
        tensor_count=len(tensors),
        tensor_elements=total_elements,
        tensor_data_bytes=total_data_bytes,
        dtype_counts=dict(sorted(dtype_counts.items())),
        tensors=tensors,
    )
