# Legacy GGML v1 (`lmgg`) file layout

This note describes the legacy GGML v1 layout recognized by `model-inspect`. It
is a readable map of the bytes in a file, not a claim that every historical
GGML-family format has the same layout.

There is no standalone, complete normative GGML v1 specification in the
references used by this project. The layout below is documented from the
legacy reader in the official
[`ggml-org/llama.cpp` GGML-to-GGUF converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_llama_ggml_to_gguf.py)
and checked against the `lmgg` model that motivated this tool. The converter
source is a reference implementation; its `master` URL can change over time.

## High-level layout

All integers are little-endian. There is no alignment padding between the
vocabulary and tensor records, or between tensor records and their payloads.

```text
┌──────────────────────────────────────────────────────────────┐
│ magic: 4 bytes = "lmgg"                                      │
├──────────────────────────────────────────────────────────────┤
│ 7 × uint32 hyperparameters (28 bytes)                        │
├──────────────────────────────────────────────────────────────┤
│ vocabulary: n_vocab × (uint32 byte_length + token bytes)     │
├──────────────────────────────────────────────────────────────┤
│ tensor record 0: metadata + raw tensor payload               │
│ tensor record 1: metadata + raw tensor payload               │
│ ...                                                          │
└──────────────────────────────────────────────────────────────┘
```

The file does not contain a separate tensor count in this layout. A reader
walks tensor records until EOF; each record's dtype and dimensions determine
how many payload bytes follow its metadata.

## Header

The first four bytes are the literal ASCII bytes `lmgg`. They are followed by
seven unsigned 32-bit little-endian integers, in this order:

| Order | Field | Meaning |
|---:|---|---|
| 1 | `n_vocab` | Number of vocabulary entries |
| 2 | `n_embd` | Embedding width |
| 3 | `n_mult` | Feed-forward size multiple |
| 4 | `n_head` | Attention head count |
| 5 | `n_layer` | Transformer layer count |
| 6 | `n_rot` | Rotary embedding dimension |
| 7 | `ftype` | Legacy model-wide file-type hint |

Thus, for this layout, the vocabulary starts at byte offset 32.

### `ftype`

`ftype` is a legacy, model-level description of the expected/general weight
type. In the old-format converter's enum, the values relevant to unquantized
GGML v1 models are:

| Value | Name | Meaning |
|---:|---|---|
| 0 | `ALL_F32` | Weights are expected to be F32 |
| 1 | `MOSTLY_F16` | Weights are expected to be mostly F16 |

“Mostly” matters: `ftype` is not a per-tensor directory and does not determine
the byte length of each tensor. Every tensor record has its own `dtype_id`,
which is what a reader must use to calculate that tensor's payload. A model can
contain a mixture of types (for example, F16 weights and some F32 tensors), so
inspect the tensor records when exact per-tensor types matter. The current
inspector reports unknown `ftype` numeric values as `UNKNOWN_<value>` rather
than rejecting the file based on this advisory field.

The upstream converter lists additional `ftype` values for later quantized
legacy formats, but those values alone do not make their tensor encodings
compatible with GGML v1. This inspector currently supports only F32 and F16
tensor payloads.

## Vocabulary

There are exactly `n_vocab` entries. Each entry is encoded as:

```text
uint32 token_byte_length
byte[token_byte_length] token
```

The length counts bytes, not Unicode characters. **GGML v1 entries do not
include a score field.** Do not parse this as the later GGMF/GGJT vocabulary
record shape, which added fields and version-specific behavior.

## Tensor records

Each tensor begins immediately after the previous token or tensor payload. Its
metadata is:

```text
uint32 ndims
uint32 name_byte_length
uint32 dtype_id
uint32 dims[ndims]
byte[name_byte_length] name
byte[payload_length] tensor_data
```

Dimensions are listed in file order; dimension 0 is the innermost dimension in
the inspector's output. The name is a byte string (the current inspector
requires valid UTF-8). There is no padding between the name and data, nor any
alignment gap before the next tensor.

For an unquantized type, the element count is the product of all dimensions.
The payload length is `element_count × bytes_per_element`. More generally,
quantized types use their type's block geometry:

```text
payload_length = (element_count / block_elements) × block_bytes
```

The count must be divisible by `block_elements`. A reader must know the exact
block geometry for a dtype before it can locate the next tensor safely.

### Types currently supported by this inspector

| `dtype_id` | Type | Elements per block | Bytes per block | Payload size |
|---:|---|---:|---:|---|
| 0 | F32 | 1 | 4 | `element_count × 4` |
| 1 | F16 | 1 | 2 | `element_count × 2` |

These are the types supported by the current reader, not an exhaustive list of
all dtype IDs ever used across GGML-family formats. Unknown IDs are rejected;
the inspector does not guess their size.

## Example of record boundaries

Suppose one tensor has `ndims = 2`, name `weight`, dimensions `[3, 2]`, and
`dtype_id = 1` (F16). Its tensor metadata occupies:

```text
12 bytes fixed fields
+ 8 bytes dimensions (2 × uint32)
+ 6 bytes name (UTF-8 bytes for "weight")
= 26 bytes metadata
```

It has six elements, so its F16 payload is `6 × 2 = 12` bytes. The next tensor
record starts exactly 12 bytes after the payload start. No 32-byte alignment
rounding is applied.

## Relationship to later formats

`lmgg` identifies the legacy GGML v1 format described here. Later related
formats use different magic/version conventions and may change vocabulary
records, quantization encodings, or tensor alignment. GGUF is a successor
format with its own explicit metadata and tensor layout; this document does
not describe GGUF. A parser should dispatch by magic/version rather than
assuming that every file called `.bin` follows this layout.

## Scope of `model-inspect`

The current implementation validates the `lmgg` magic, reads this seven-field
header and the score-free vocabulary, then walks tensors to EOF while seeking
over payloads. It recognizes only F32 and F16 payload sizes. It does not load
weights, validate that tensor names/shapes match a particular model
architecture, or establish model provenance or weight correctness.
