"""Command-line interface for model-inspect."""

from __future__ import annotations

import argparse
import json
import sys

from .ggml_v1 import GGMLFormatError, Inspection, inspect


def _print_text(result: Inspection, show_tensors: bool) -> None:
    hp = result.hyperparameters
    print(f"File: {result.path}")
    print(f"Format: {result.format} v{result.format_version} (ftype: {result.ftype})")
    print(f"Size: {result.file_size:,} bytes")
    print(
        "Architecture: "
        f"vocab={hp['n_vocab']:,}, embedding={hp['n_embd']:,}, "
        f"layers={hp['n_layer']}, heads={hp['n_head']}, rotary={hp['n_rot']}"
    )
    print(f"Tensors: {result.tensor_count} ({', '.join(f'{k}={v}' for k, v in result.dtype_counts.items())})")
    print(f"Tensor elements: {result.tensor_elements:,}")
    print(f"Tensor payload: {result.tensor_data_bytes:,} bytes")

    if show_tensors:
        print("\nTensors (shape uses file order; dim 0 is the innermost dimension):")
        for tensor in result.tensors:
            shape = " × ".join(str(dim) for dim in tensor.shape)
            print(f"{tensor.name}\t{shape}\t{tensor.dtype}\t{tensor.data_bytes:,} bytes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model-inspect",
        description="Read-only metadata inspector for legacy GGML v1 model files.",
    )
    parser.add_argument("model_file", help="path to a legacy GGML v1 file (magic: lmgg)")
    parser.add_argument("--json", action="store_true", help="emit JSON including tensor metadata")
    parser.add_argument("--tensors", action="store_true", help="list tensors in text output")
    args = parser.parse_args(argv)

    try:
        result = inspect(args.model_file)
    except (OSError, GGMLFormatError) as exc:
        print(f"model-inspect: {exc}", file=sys.stderr)
        return 2

    if args.json:
        json.dump(result.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        _print_text(result, args.tensors)
    return 0
