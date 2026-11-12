# Upstream fact sheet — GGUF format

Every value below was read from llama.cpp source on 2026-11-05, not recalled.
Anything not verified is marked **待查证**.

- Repo: `https://github.com/ggml-org/llama.cpp`
- Ref: `master` (sliding ref) — pinned commit `186eab1cb83899282bec44b837270959b56ee8f3`, fetched 2026-11-05T21:16:56Z
- Re-check method: `curl -sS https://raw.githubusercontent.com/ggml-org/llama.cpp/master/<path> | sha256sum`

## Files and digests

| File | sha256 (first 16) | lines |
|---|---|---|
| `gguf-py/gguf/constants.py` | `53e6dfa95083d5e8` | 7156 |
| `gguf-py/gguf/gguf_reader.py` | `d0ea743200e19d7e` | 390 |
| `gguf-py/gguf/gguf_writer.py` | `a5c910b3a7181b0b` | 1898 |
| `ggml/src/gguf.cpp` | `51bd966ab0494aa2` | 1755 |
| `ggml/include/gguf.h` | `e56714aab702e5ce` | — |
| `ggml/include/ggml.h` | `590e00a522b87912` | 3084 |
| `ggml/src/ggml-common.h` | `0061131b615c5721` | 1911 |

Digests of the first four were independently recomputed by the coordinator and matched.

## Header constants

| Constant | Value | Source |
|---|---|---|
| `GGUF_MAGIC` | `0x46554747` (`"GGUF"` LE) | `constants.py:10`; `gguf.h:41` (string form) |
| `GGUF_VERSION` | `3` | `constants.py:11`; `gguf.h:42` |
| `GGUF_DEFAULT_ALIGNMENT` | `32` | `constants.py:12`; corroborated `gguf.cpp:225` |
| `GGUF_KEY_GENERAL_ALIGNMENT` | `"general.alignment"` | `gguf.h:44`; `constants.py:26` (`Keys.General.ALIGNMENT`) |
| `GGML_MAX_DIMS` | `4` | `ggml.h:222` |
| `GGML_MAX_NAME` | `64` | `ggml.h:229` |
| `GGUF_MAX_STRING_LENGTH` | `1<<30` | `gguf_reader.py:37`; `gguf.cpp:20` |
| `GGUF_MAX_ARRAY_ELEMENTS` | `1<<30` | `gguf_reader.py:38`; `gguf.cpp:21` |
| `READER_SUPPORTED_VERSIONS` | `[2, 3]` | `gguf_reader.py:41` |

**Note on `GGUF_KEY_GENERAL_ALIGNMENT`**: it is a C macro string defined in `gguf.h:44`. The
earlier "待查证" flag is resolved — exact spelling is `general.alignment`.

## Header wire layout

`gguf_writer.py:229-235` (`write_header_to_file`):

```
"<I"  magic          (4 bytes)
"I"   version        (4 bytes)
"Q"   tensor_count   (8 bytes)
"Q"   kv_count       (8 bytes)
```

`gguf.cpp:520-551` reads the same: `read(version)` as `uint32`, then `n_tensors` and `n_kv`
both as `int64`.

## GGUF v2 vs v3 — VERIFIED: no wire difference in what this tool reads

This was the open question. Finding:

- The C++ reader (`gguf.cpp:493-519`) treats versions uniformly after parsing: it rejects
  `version == 0`, rejects `version == 1`, and rejects `version > GGUF_VERSION (3)`. There is
  **no branch that changes field widths based on version**. The only version-dependent code is
  the endianness sniff (`version & 0x0000FFFF == 0`).
- The Python reader (`gguf_reader.py:147-167`) reads version as `np.uint32`, then immediately
  reads tensor_count and kv_count as `np.uint64` (`:170`) with no version branch.
- `gguf_writer.py` only ever writes `GGUF_VERSION` (3); it cannot produce v2, so there is no
  writer-side version branch to consult.

**Historical context, not verified from current source**: the commonly cited difference is that
GGUF v1 used `uint32` for tensor_count/kv_count and `uint32` string lengths, and v2 moved both
to `uint64`. Current upstream no longer supports v1 at all, so the v1 layout is not present in
the code above. Treat "v2 = v3 layout" as the operative fact for this tool. **待查证** if we
ever need v1.

**Consequence for the SPEC**: supporting v2 means accepting `version ∈ {2, 3}` and parsing both
identically. It is cheap. Reject 0, 1, and >3 with distinct errors.

## Endianness

`gguf.cpp:498-509` — if `(version & 0x0000FFFF) == 0`, upstream reports a likely host/model
endianness mismatch. `gguf_reader.py:148-152` uses the same test to set `byte_order = 'S'`
(swapped) and continues. Magic is always read little-endian (`gguf_reader.py:142`,
`override_order = '<'`).

Decision (human-approved): reject big-endian in phase one with a clear error, rather than
supporting byte-swapped parsing.

## KV pairs

`gguf.cpp:556-617`:

1. key (string), rejected if empty, rejected if duplicate
2. type (`int32`, read as `gguf_type`)
3. if `type == ARRAY`: element type (`int32`) then count (`uint64`)
4. value(s) per type

`GGUFValueType` (`constants.py`):

| id | name | id | name |
|---|---|---|---|
| 0 | UINT8 | 7 | BOOL |
| 1 | INT8 | 8 | STRING |
| 2 | UINT16 | 9 | ARRAY |
| 3 | INT16 | 10 | UINT64 |
| 4 | UINT32 | 11 | INT64 |
| 5 | INT32 | 12 | FLOAT64 |
| 6 | FLOAT32 | | |

BOOLEAN wire encoding: `gguf.cpp:313-321` reads a single `int8`, non-zero = true.

String encoding: length as `uint64`, then that many bytes (not NUL-terminated).

## Alignment

`gguf.cpp:622-638`:
- if `general.alignment` present, it **must** be `UINT32`; otherwise hard error
- default `GGUF_DEFAULT_ALIGNMENT = 32`
- must be non-zero and a power of two; otherwise hard error

Data section start: after tensor info, seek to `GGML_PAD(current, alignment)` (`gguf.cpp:781`).

## Tensor info

`gguf_writer.py:263` (write) and `gguf.cpp:642-771` / `gguf_reader.py:282-296` (read):

```
name    (string)
n_dims  (uint32)
dims    (uint64 × n_dims)   -- stored so that ne[0] is the innermost/most minor dim;
                               writer reverses the caller's shape on write, reader reverses on read
dtype   (uint32, ggml_type)
offset  (uint64)            -- byte offset relative to start of data section
```

Validation performed by upstream `gguf.cpp` that the Python reader does not do:
- tensor name length `>= GGML_MAX_NAME (64)` → error (`:657`)
- duplicate tensor name → error (`:665`)
- `n_dims > GGML_MAX_DIMS (4)` → error (`:679`)
- negative dimension → error
- element-count overflow (`INT64_MAX`) → error (`:700-720`)
- `dtype` outside `[0, GGML_TYPE_COUNT)` → error (`:723`)
- `ne[0] % blck_size != 0` → error (`:733`)
- byte-size overflow → error (`:741`)

Python reader only checks duplicate tensor names (`gguf_reader.py:342`). **Divergence is real.**
Our Go implementation should follow the C++ reader (stricter), and the SPEC must say so.

## GGML tensor types

Derived by evaluating the `static_assert(sizeof(block_*) == ...)` formulas in `ggml-common.h`
together with `#define QK_*` in the same file, and the enum order in `ggml.h:391+`.

**Trap, documented by the earlier investigator and confirmed here**: `type_traits` in `ggml.c`
uses `sizeof(block_q*)`, not numeric literals. A generator that regexes `ggml.c` for numbers will
produce silently wrong output. The generator must parse `ggml-common.h`.

id / name / block_elems / block_bytes:

```
F32      0/1/4      F16      1/1/2      Q4_0     2/32/18    Q4_1     3/32/20
Q5_0     6/32/22    Q5_1     7/32/24    Q8_0     8/32/34    Q8_1     9/32/36
Q2_K    10/256/84   Q3_K    11/256/110  Q4_K    12/256/144  Q5_K    13/256/176
Q6_K    14/256/210  Q8_K    15/256/292  IQ2_XXS 16/256/66   IQ2_XS  17/256/74
IQ3_XXS 18/256/98   IQ1_S   19/256/50   IQ4_NL  20/32/18    IQ3_S   21/256/110
IQ2_S   22/256/82   IQ4_XS  23/256/136  I8      24/1/1      I16     25/1/2
I32     26/1/4      I64     27/1/8      F64     28/1/8      IQ1_M   29/256/56
BF16    30/1/2      TQ1_0   34/256/54   TQ2_0   35/256/66   MXFP4   39/32/17
NVFP4   40/64/36    Q1_0    41/128/18   Q2_0    42/64/18    F8_E4M3 43/1/1
```

`GGML_TYPE_COUNT = 44`. Ids `4, 5, 31, 32, 33, 36, 37, 38` are removed/reserved and must not be
assigned. Supporting `QK_*` values found in `ggml-common.h`: `QK_K=256` (:89), `QK1_0=128`
(:180), `QK2_0=64` (:187), `QK4_0=32` (:194), `QK4_1=32` (:201), `QK_MXFP4=32` (:214),
`QK_NVFP4=64` (:221), `QK_NVFP4_SUB=16` (:222).

**Not yet machine-verified**: this table was hand-transcribed from the earlier run's computed
output and spot-checked, but the per-type `block_bytes` values have not been re-derived by an
independent script in this session. The generator in `internal/gguf/gen/` must reproduce this
table from upstream, and a test must compare the two. Until that test exists, treat the table as
provisional.

## Other facts

- `LLAMA_FILE_TYPE` mapping (`general.file_type`) is a **separate** enum from `ggml_type`; it was
  handled in the archived Python version and is not in scope for phase one unless required.
- Upstream `gguf_dump.py` JSON shape has no schema version field; we define our own.

## Explicitly unverified

1. Per-type `block_bytes` for all 38 entries (see above) — needs generator cross-check.
2. Actual byte offsets round-tripped against a real `.gguf` file. No `.gguf` exists on this host.
3. GGUF v1 layout (only relevant if we ever claim v1 support; we do not).
4. Whether any divergence exists between `gguf.cpp` and `gguf_reader.py` on zero-element tensors.
   `gguf.cpp:701` comment says a zero-element tensor is trivially representable, but the
   `ne[0] % blck_size != 0` check at `:733` would reject `ne[0] == 0` only if blck_size were 0.
   Behaviour for `ne[0] == 0` with a quantized dtype is unprobed.

## Provenance update (2026-11-05, 22:44 UTC)

The type table above was independently re-derived by a separate agent using three methods
(Python C-layout computation, gcc `sizeof` probe against the real header, evaluation of all 27
upstream `static_assert`s) and cross-checked against `GGML_QUANT_SIZES` in `constants.py`.

Result: **44/44 rows identical to the table in this file.** See
`internal/gguf/docs/ggml-types.md` and `internal/gguf/docs/derived_output.txt`.

The "未重推导 / not yet machine-verified" caveat in this document is now **resolved**. The
authoritative table is `internal/gguf/docs/ggml-types.md`, not this file.

Three upstream traps found during that verification, which the generator must handle:

1. `ggml-common.h` lives at `ggml/src/ggml-common.h`, **not** at the repo root. The root path
   returns an HTTP 404 whose body is the literal string `404: Not Found` — a naive fetch
   silently saves garbage.
2. `static_assert(sizeof(block_*) == ...)` alone does not prove the size. Six structs
   (`q4_1`, `q5_1`, `q8_1`, `q2_K`, `q4_K`, `q5_K`) contain a `half2 dm` union and depend on
   alignment padding. An incorrect layout model (no union alignment) produced 24/28/40/88/148/180
   instead of 20/24/36/84/144/176 and still "passed" the asserts. Proof must come from a compiler.
3. Macro and struct names do not map mechanically: `QK_MXFP4` → `block_mxfp4`,
   `QK4_NL` → `block_iq4_nl`, and 11 IQ/TQ types have no `QK<name>` macro at all. A
   `QK<NAME>` lookup silently misses or mismatches them.
