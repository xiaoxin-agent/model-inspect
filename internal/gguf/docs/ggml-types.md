# GGML tensor types — independently derived `block_elems` / `block_bytes`

Independent verification for `model-inspect`. Every number below comes from the scripts in
this directory run against freshly downloaded upstream files. `SPEC-NOTES.md` was **not**
used as a source (it was only opened afterwards to see which claims exist; its values are
not quoted here as evidence).

- Derived: 2026-11-05 (UTC)
- Ref: `ggml-org/llama.cpp` `master` (sliding ref), fetched **twice** during the session,
  same digests both times.
- Files fetched by this session (not a local copy):

| Upstream path | sha256 | bytes | local copy |
|---|---|---:|---|
| `ggml/include/ggml.h` | `590e00a522b8791235353a8c7388d67ff2fe929554f1834c58528b82afd67392` | 3084 lines | `internal/gguf/docs/upstream/ggml.h` |
| `ggml/src/ggml-common.h` | `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c` | 1911 lines | `internal/gguf/docs/upstream/ggml-common.h` |
| `gguf-py/gguf/constants.py` | `53e6dfa95083d5e81d1da316beb9ea1d4f943cec9dd450db54d849023c01df6c` | 7156 lines | `internal/gguf/docs/upstream/constants.py` |

URLs:

```
https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h
https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h
https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/constants.py
```

Note: `ggml-common.h` is **not** at the repo root; the root path returns `404: Not Found`.
The file lives at `ggml/src/ggml-common.h`.

## Method (three independent sources)

1. **Enum order** — parsed `enum ggml_type { ... }` in `ggml.h:391-437`. Removed ids are
   present only as comments, so the parser reproduces the gaps rather than assuming them.
2. **block_elems** — from `#define QK_*` / `#define QK*` in `ggml-common.h`:
   `QK_K 256` (:89), `QK1_0 128` (:183), `QK2_0 64` (:187), `QK4_0 32`, `QK4_1 32`,
   `QK_MXFP4 32` (:214), `QK_NVFP4 64` (:221), `QK5_0/5_1 32`, `QK8_0/8_1 32`,
   `QK4_NL 32`, and `QK_K` for every `*_K` / IQ-`/TQ-` super-block. For the 11 block
   types whose names do not map mechanically to a macro (they have no `QK<name>` define —
   e.g. `block_iq2_xxs`, `block_tq1_0`), `derive_types.py` uses an explicit table
   `TYPE_TO_BLOCK_SUFFIX` plus `QK_K`, and the script self-checks that each entry names an
   existing `block_*` struct. No value was assumed for these: each is pinned by its
   upstream `static_assert` which is written in terms of `QK_K`.
3. **block_bytes** — two computations, cross-checked:
   - `derive_types.py` parses each `typedef struct {...} block_x;` and computes the C
     layout (member alignment, tail padding). `ggml-common.h` has no `#pragma pack`; under
     `GGML_COMMON_DECL_C/_CPP` `ggml_half = uint16_t` (2 B) and `ggml_half2 = uint32_t` (4 B).
   - `probe_sizes.c` is compiled with `gcc -std=c11` against the real header and prints
     `sizeof`/`_Alignof` for every block. **27/27 identical to the Python layout.**
   - Every one of the 27 upstream `static_assert(sizeof(block_*) == ...)` expressions was
     also evaluated by the script: **27/27 agree** with the computed size.
4. **Third party** — `GGML_QUANT_SIZES` in `constants.py:7035-7072` parsed and compared.

Reproduce:

```bash
cd internal/gguf/docs
gcc -std=c11 -o /tmp/probe probe_sizes.c -I. && /tmp/probe > /tmp/probe.txt
python3 derive_types.py --gcc-probe /tmp/probe.txt   # full output incl. per-member layout
```

Full raw output of the run: `internal/gguf/docs/derived_output.txt`.

## Table

`GGML_TYPE_COUNT = 44` (`ggml.h:436`). All 44 ids 0–43 are listed; 8 ids have no enum
entry and are reserved.

| id | name | block_elems | block_bytes |
|---:|---|---:|---:|
| 0 | F32 | 1 | 4 |
| 1 | F16 | 1 | 2 |
| 2 | Q4_0 | 32 | 18 |
| 3 | Q4_1 | 32 | 20 |
| 4 | *(reserved — `Q4_2` removed)* | – | – |
| 5 | *(reserved — `Q4_3` removed)* | – | – |
| 6 | Q5_0 | 32 | 22 |
| 7 | Q5_1 | 32 | 24 |
| 8 | Q8_0 | 32 | 34 |
| 9 | Q8_1 | 32 | 36 |
| 10 | Q2_K | 256 | 84 |
| 11 | Q3_K | 256 | 110 |
| 12 | Q4_K | 256 | 144 |
| 13 | Q5_K | 256 | 176 |
| 14 | Q6_K | 256 | 210 |
| 15 | Q8_K | 256 | 292 |
| 16 | IQ2_XXS | 256 | 66 |
| 17 | IQ2_XS | 256 | 74 |
| 18 | IQ3_XXS | 256 | 98 |
| 19 | IQ1_S | 256 | 50 |
| 20 | IQ4_NL | 32 | 18 |
| 21 | IQ3_S | 256 | 110 |
| 22 | IQ2_S | 256 | 82 |
| 23 | IQ4_XS | 256 | 136 |
| 24 | I8 | 1 | 1 |
| 25 | I16 | 1 | 2 |
| 26 | I32 | 1 | 4 |
| 27 | I64 | 1 | 8 |
| 28 | F64 | 1 | 8 |
| 29 | IQ1_M | 256 | 56 |
| 30 | BF16 | 1 | 2 |
| 31 | *(reserved — `Q4_0_4_4` removed)* | – | – |
| 32 | *(reserved — `Q4_0_4_8` removed)* | – | – |
| 33 | *(reserved — `Q4_0_8_8` removed)* | – | – |
| 34 | TQ1_0 | 256 | 54 |
| 35 | TQ2_0 | 256 | 66 |
| 36 | *(reserved — `IQ4_NL_4_4`)* | – | – |
| 37 | *(reserved — `IQ4_NL_4_8`)* | – | – |
| 38 | *(reserved — `IQ4_NL_8_8`)* | – | – |
| 39 | MXFP4 | 32 | 17 |
| 40 | NVFP4 | 64 | 36 |
| 41 | Q1_0 | 128 | 18 |
| 42 | Q2_0 | 64 | 18 |
| 43 | F8_E4M3 | 1 | 1 |

**Rows: 44 ids total — 36 defined types + 8 reserved. `GGML_TYPE_COUNT = 44`.**

Enums with an explicit trailing comma and no gaps not listed above: none. The only ids
absent from `ggml.h` are 4, 5, 31, 32, 33, 36, 37, 38.

Raw (non-block) types F32/F16/I8/I16/I32/I64/F64/BF16/F8_E4M3 have block_elems 1; their
sizes are the scalar widths per `GGML_QUANT_SIZES` and agree with the byte-size of the C
types. They are the only rows whose bytes do **not** come from a `block_*` struct.

## Comparison with `constants.py` `GGML_QUANT_SIZES`

**Result: 36/36 defined types agree exactly (elems and bytes). 0 differences.**
The dict has no entry for the 8 reserved ids (expected — they are not types).

The registry `GGMLQuantizationType` (`constants.py:6802-6838`) has exactly the same ids and
names as `ggml.h` (no id differs, no extra/missing member).

### Anomalies found (not discrepancies in the numbers)

1. **Upstream `static_assert`s are not a complete check of struct size.** Six asserts omit
   tail padding from the union member `half2 dm` (4-byte align):
   `block_q4_1`, `block_q5_1`, `block_q8_1`, `block_q2_K`,
   `block_q4_K`, `block_q5_K`. During development my first, deliberately naive layout model
   (ignoring the union's `uint32_t` alignment) produced 24/28/40/88/148/180 for these and
   the asserts "agreed" with the wrong numbers — a useful negative control showing the
   asserts alone are a weaker oracle than compiling. After modelling the union correctly,
   computed size == assert == gcc for all 27. Conclusion: the asserts happen to be written
   as `2*sizeof(ggml_half) + ...` which equals the padded struct size; they are correct
   here, but they are not what proves it.
2. **Name mismatch between macro and struct** for Q1_0/Q2_0/MXFP4/NVFP4/IQ4_NL:
   macro spellings `QK_MXFP4`, `QK_NVFP4`, `QK4_NL` vs structs `block_mxfp4`,
   `block_nvfp4`, `block_iq4_nl`. A naive `QK<NAME>` lookup silently misses these; any
   generator must not assume the macro name derives from the type name.
3. **`ggml-common.h` is not at repo root.** A tool that fetches
   `raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml-common.h` gets a 404 page
   (14 bytes of text) with HTTP 200-less status handling risks. Use `ggml/src/ggml-common.h`.
   I hit this on the first fetch and discarded that file.
4. **`type_traits` in `ggml.c` uses `sizeof(block_*)`**, not literals — as instructed, not
   used as a source. Consequence: ggml's runtime `type_size` for these types equals the
   struct size including padding, which is what the table reports.

## Risks / unknowns

- `master` is a sliding ref. Digests above pin this run; if upstream changes a block layout
  the table must be re-derived (rerun the two commands).
- Q1_0 (128 elems) and Q2_0 (64 elems) are newer than most readers; a consumer must not
  assume every quant is 32 or 256 elements.
- NVFP4 blocks are 64 elements with 4 sub-block scales; MXFP4 is 32 elements with a single
  E8M0 byte and no half scale — different layout shape from Q4_0 despite similar names.
- Not checked here: whether `GGUF` writers enforce `ne[0] % block_elems == 0`, which is a
  separate question from this table.

## Checks performed

- `curl` of the three upstream URLs; sha256 + line counts recorded.
- `python3 derive_types.py --gcc-probe /tmp/probe.txt` → 27 static_assert checks OK,
  27/27 gcc vs Python layout OK, 36/36 match vs `constants.py`, 0 diff.
- `gcc -std=c11` compile of `probe_sizes.c` against the real header (exit 0).
- `diff` of two independently built probe binaries (`/tmp/probe`, `/tmp/probe2`): identical.
- Files written: `derive_types.py`, `probe_sizes.c`, `derived_output.txt`, this file,
  plus `upstream/` copies. Nothing committed.
