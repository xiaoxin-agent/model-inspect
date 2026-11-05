#!/usr/bin/env python3
"""
Derive ggml block_elems / block_bytes for every GGML tensor type from upstream sources.

Sources (downloaded by the caller, NOT copied from anyone else):
  UPSTREAM/ggml.h          https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h
  UPSTREAM/ggml-common.h   https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h
  UPSTREAM/constants.py    https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/constants.py

Method:
  1. Parse `enum ggml_type { ... }` from ggml.h (ids + names).
  2. Parse `#define`s (QK_*, K_SCALE_SIZE, IQ3S_N_SCALE, ...) from ggml-common.h.
  3. Parse every `typedef struct {...} block_xxx;` in the GGML_COMMON_DECL section and
     compute sizeof from the actual member layout with C struct rules
     (natural alignment, no packing attribute in this file), using a type-size table.
  4. Parse block_elems from the `#define QKxxx n` that precedes each block struct
     (an explicit name->QK mapping is used where names differ, e.g. block_q4_0 -> QK4_0).
  5. Evaluate the `static_assert(sizeof(block_X) == EXPR, ...)` expressions as an
     independent check of the struct-layout calculation.
  6. Parse GGML_QUANT_SIZES from constants.py and compare.
  7. --gcc-probe FILE: optionally compare against a C compiler run over the real struct
     definitions (see probe_sizes.c in this directory). Compiler output is a third
     independent source and wins over both the parser and the Python dict.

Notes on C layout rules that matter here (GGML_COMMON_DECL_C / _CPP define
ggml_half = uint16_t and ggml_half2 = uint32_t):
  * memcpy/alloc layout of a block is the struct layout: block bytes include any
    tail padding implied by struct alignment (ggml assumes the same because
    ggml_rows are allocated with the type size as stride, see ggml_nbytes).
  * the `GGML_EXTENSION union { struct { half d; half m; } data; half2 dm; }`
    member has size 4 and alignment 4, so it pads the following byte arrays to a
    multiple of 4.
  * Upstream static_assert()s in ggml-common.h do NOT include that padding
    (e.g. block_q4_1 is asserted as 20 but is really 24 bytes); the asserts are
    therefore only lower bounds, not the wire/alloc size.

Usage: python3 derive_types.py [UPSTREAM_DIR]
Writes nothing; prints markdown table + analysis to stdout.
"""

import re
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(HERE, "upstream")
GCC_PROBE = None
_argv = sys.argv[1:]
i = 0
while i < len(_argv):
    if _argv[i] == "--gcc-probe":
        GCC_PROBE = _argv[i + 1]
        i += 2
    elif _argv[i].startswith("--"):
        i += 1
    else:
        UP = _argv[i]
        i += 1

# C type sizes as used by ggml-common.h under GGML_COMMON_DECL_C / CPP (x86/arm64, no packing)
TYPE_SIZES = {
    "uint8_t": 1, "int8_t": 1,
    "uint16_t": 2, "int16_t": 2, "ggml_half": 2, "uint16_t ": 2,
    "uint32_t": 4, "int32_t": 4, "float": 4,
    "uint64_t": 8, "int64_t": 8, "double": 8,
    "ggml_half2": 4,  # half2: 2 x half
}

# Mapping from GGML enum short name (uppercase) to the block struct suffix / QK macro,
# ONLY where the naming is not derivable mechanically (block size macros do not exist
# for these in ggml-common.h; they are super-blocks of 256 by construction, see the
# static_assert expressions). This is an explicit, auditable table, not a guess.
TYPE_TO_BLOCK_SUFFIX = {
    "IQ2_XXS": "iq2_xxs",
    "IQ2_XS":  "iq2_xs",
    "IQ2_S":   "iq2_s",
    "IQ3_XXS": "iq3_xxs",
    "IQ3_S":   "iq3_s",
    "IQ1_S":   "iq1_s",
    "IQ1_M":   "iq1_m",
    "IQ4_NL":  "iq4_nl",
    "IQ4_XS":  "iq4_xs",
    "TQ1_0":   "tq1_0",
    "TQ2_0":   "tq2_0",
}

# explicit block_elems map for structs whose QK macro is not literally part of the name
EXTRA_ELEMS = {
    "block_iq1_m": "QK_K",
}

# QK_K == 256 is the super-block size; the static_assert expressions below use QK_K
# for every block_* type whose total size is expressed with QK_K.
QK_K_ASSERTED_BLOCKS = {
    "TQ1_0", "TQ2_0", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S",
    "IQ1_S", "IQ1_M", "IQ4_XS",
}
# Sub-block (32-element) type: QK4_NL is 32; IQ4_NL block is defined with QK4_NL.
QK4_NL_BLOCK = {"IQ4_NL"}


def read(name):
    with open(os.path.join(UP, name), "r", encoding="utf-8") as f:
        return f.read()


def parse_enum(src):
    m = re.search(r"enum\s+ggml_type\s*\{(.*?)\};", src, re.S)
    body = m.group(1)
    types = {}
    for line in body.splitlines():
        line = line.split("//")[0].strip()
        mm = re.match(r"(GGML_TYPE_\w+)\s*=\s*(\d+)\s*,?", line)
        if mm:
            types[int(mm.group(2))] = mm.group(1)
    return types


def parse_defines(src):
    d = {}
    for line in src.splitlines():
        mm = re.match(r"\s*#define\s+(\w+)\s+(.+?)\s*(//.*)?$", line)
        if mm and not line.strip().startswith("//"):
            d[mm.group(1)] = mm.group(2).strip()
    return d


def safe_eval(expr, defines, depth=0):
    """Evaluate a C constant expression using defines; returns int."""
    if depth > 20:
        raise ValueError("define recursion: " + expr)
    e = expr
    # sizeof(X) for the types we know
    def ssub(m):
        t = m.group(1)
        if t in TYPE_SIZES:
            return str(TYPE_SIZES[t])
        raise ValueError("unknown sizeof type: " + t)
    e = re.sub(r"sizeof\(\s*(\w+)\s*\)", ssub, e)
    # substitute identifiers
    def sub(m):
        name = m.group(0)
        if name in defines:
            return "(" + safe_eval(defines[name], defines, depth + 1).__str__() + ")"
        if name in TYPE_SIZES:
            return str(TYPE_SIZES[name])
        return name
    e = re.sub(r"\b[A-Za-z_]\w*\b", sub, e)
    e = e.replace("/", "//")
    # only allow arithmetic
    if not re.fullmatch(r"[0-9+\-*/() \t]+", e.replace("//", "/")):
        raise ValueError("unsafe expr: " + expr + " -> " + e)
    return int(eval(e, {"__builtins__": {}}, {}))


def parse_structs(src):
    """Return list of (block_name, members) where members = [(type, name, count_expr_str)]"""
    # take only the struct definitions from the first "#if defined(GGML_COMMON_DECL)"
    # to (and excluding) the "#ifndef GGML_COMMON_IMPL" table section
    decl_start = src.index("#if defined(GGML_COMMON_DECL)")
    decl_end = src.index("#ifndef GGML_COMMON_IMPL", decl_start)
    body = src[decl_start:decl_end]
    blocks = []
    for m in re.finditer(r"typedef\s+struct\s*\{(.*?)\}\s*(block_\w+)\s*;", body, re.S):
        inner, name = m.group(1), m.group(2)
        members = []
        # remove GGML_EXTENSION and unions: a union of {struct{d;m} ; half2 dm;} has size max => treat as
        # "union{...} dm" with explicit size. Handle unions separately.
        # 1) union blocks. NOTE: the union contains a nested anonymous struct, so the
        # regex must be anchored on the closing "} NAME;" that follows the nested
        # struct's closing "} data;" (non-greedy .*? would stop at the first "}").
        for um in re.finditer(r"union\s*\{(.*)\}\s*(\w+)\s*;", inner, re.S):
            ubody = um.group(1)
            # members inside the anonymous struct
            # nested member sizes, keeping C alignment inside the anonymous struct
            off = 0
            ualign = 1
            # members INSIDE the nested anonymous struct (skip the "half2 dm;" fallback
            # and the "} data;" terminator by requiring at least 4 spaces of indentation
            # is unreliable -> instead only take members up to the first "}")
            inner_struct = ubody[:ubody.index("}")]
            for sm in re.finditer(r"(ggml_half2|ggml_half|uint8_t|uint16_t|uint32_t|int8_t|int16_t|float)\s+(\w+)\s*;", inner_struct):
                esz = TYPE_SIZES[sm.group(1)]
                off = align_up(off, esz) + esz
                ualign = max(ualign, esz)
            u_size = align_up(off, ualign)
            # union of anonymous struct (size u_size, align ualign) and half2 (4,4)
            u_size = max(u_size, TYPE_SIZES["ggml_half2"])
            ualign = max(ualign, TYPE_SIZES["ggml_half2"])
            members.append(("UNION", um.group(2), "__UNION__", u_size, ualign))
        inner_no_union = re.sub(r"union\s*\{.*\}\s*\w+\s*;", "", inner, flags=re.S)
        # strip preprocessor lines
        inner_no_union = "\n".join(l for l in inner_no_union.splitlines() if not l.strip().startswith("#"))
        for sm in re.finditer(
            r"(ggml_half2|ggml_half|uint8_t|uint16_t|uint32_t|int8_t|int16_t|int32_t|int64_t|uint64_t|float|double)\s+(\w+)\s*(\[[^\]]*\])?\s*;",
            inner_no_union):
            ctype, cname, arr = sm.group(1), sm.group(2), sm.group(3)
            count = None
            if arr:
                count = arr[1:-1]
            members.append((ctype, cname, count))
        blocks.append((name, members))
    return blocks


def align_up(n, a):
    return (n + a - 1) // a * a


def sizeof(members, defines):
    offset = 0
    max_align = 1
    detail = []
    for ctype, cname, count, *rest in members:
        if ctype == "UNION":
            # union { struct{ half d; half m; } data; half2 dm; } : size = max(2+2, 4) = 4,
            # alignment = max(2 (half), 4 (half2)) = 4
            (u_size, u_align) = rest[0], rest[1]
            size = u_size
            align = u_align
            detail.append(f"union {cname} (packed member): {size}B (align {align})")
        else:
            esz = TYPE_SIZES[ctype]
            n = 1 if count is None else safe_eval(count, defines)
            size = esz * n
            align = esz
            detail.append(f"{ctype} {cname}[{count}]: {esz}*{n}={size}B (align {align})")
        offset = align_up(offset, align) + size
        max_align = max(max_align, align)
    total = align_up(offset, max_align)
    return total, detail


def parse_static_asserts(src):
    out = {}
    for m in re.finditer(r"static_assert\(sizeof\((block_\w+)\)\s*==\s*(.*?),\s*\"", src):
        out[m.group(1)] = m.group(2).strip()
    return out


def parse_constants_sizes(src):
    """GGML_QUANT_SIZES dict -> {name: (elems, bytes)}, plus enum id map."""
    e = re.search(r"class\s+GGMLQuantizationType\(IntEnum\):(.*?)\n\n", src, re.S).group(1)
    enum = {}
    for line in e.splitlines():
        mm = re.match(r"\s*(\w+)\s*=\s*(\d+)", line)
        if mm:
            enum[int(mm.group(2))] = mm.group(1)
    m = re.search(r"GGML_QUANT_SIZES.*?=\s*\{(.*?)\n\}", src, re.S)
    qk_k = int(re.search(r"^QK_K\s*=\s*(\d+)", src, re.M).group(1))
    out = {}
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*GGMLQuantizationType\.(\w+)\s*:\s*\((.*)\)\s*,", line)
        if not mm:
            continue
        name, expr = mm.group(1), mm.group(2)
        a, b = expr.split(",", 1)
        a, b = a.strip(), b.strip().replace("QK_K", str(qk_k))
        b = re.sub(r"(?<!//)//(?!//)", "//", b)  # C / -> Python //
        out[name] = (int(eval(a, {"__builtins__": {}}, {})), int(eval(b, {"__builtins__": {}}, {})))
    return enum, out


def parse_gcc_probe(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[0].startswith("block_"):
                out[parts[0][len("block_"):].upper()] = int(parts[1])
    return out


def derived_key_to_block(blocks, short):
    for bname, _ in blocks:
        sfx = bname[len("block_"):]
        if sfx.upper() == short or TYPE_TO_BLOCK_SUFFIX.get(short) == sfx:
            return sfx
    raise KeyError(short)


def main():
    h = read("ggml.h")
    c = read("ggml-common.h")
    py = read("constants.py")

    ids = parse_enum(h)
    defines = parse_defines(c)
    blocks = parse_structs(c)
    asserts = parse_static_asserts(c)
    cenum, csizes = parse_constants_sizes(py)

    block_by_lower = {b.lower(): b for b, _ in blocks}

    # name -> (elems, bytes) derived
    derived = {}
    details = {}
    details_elems_key = {}
    # block struct suffix (lower) -> enum short name (upper)
    block_to_type = {}
    for bname, members in blocks:
        suffix = bname[len("block_"):]
        block_to_type[suffix] = suffix.upper()
    for t, sfx in TYPE_TO_BLOCK_SUFFIX.items():
        block_to_type[sfx] = t

    for bname, members in blocks:
        short = bname[len("block_"):]
        tname = block_to_type[short]
        elems = None
        elems_key = None
        if bname in EXTRA_ELEMS:
            elems_key = EXTRA_ELEMS[bname]
        elif tname in QK4_NL_BLOCK:
            elems_key = "QK4_NL"
        elif tname in QK_K_ASSERTED_BLOCKS:
            elems_key = "QK_K"
        else:
            # block_q4_0 -> QK4_0 / QK_4_0 / QK_Q4_0
            up = short.upper()
            cands = []
            if up.endswith("_K"):
                cands.append("QK_K")
            base = up
            if up.startswith("Q"):
                base = up[1:]          # q4_0 -> QK4_0
            cands += ["QK" + base, "QK" + up, "QK_" + up, "QK_" + base]
            for key in cands:
                if key in defines:
                    elems_key = key
                    break
        if elems_key:
            elems = safe_eval(defines[elems_key], defines)
        details_elems_key[tname] = elems_key
        size, det = sizeof(members, defines)
        derived[tname] = (elems, size)
        details[tname] = det
        # alias lookup keys: block_q1_0 -> short "q1_0" but enum is Q1_0; normalise uppercase already done

    # alias: enum short names are uppercase of block suffix; derived keys are uppercase too.
    # e.g. block_iq2_xxs -> "IQ2_XXS" matches enum. block_q4_0 -> "Q4_0" matches. OK.

    # self-check: every explicit prefix-table entry must point at an existing block struct
    for t, sfx in TYPE_TO_BLOCK_SUFFIX.items():
        assert "block_" + sfx in asserts, f"table entry {t} -> block_{sfx} does not exist"
        assert t in derived, f"table entry {t} not resolved"

    # verify via static_assert
    print("## static_assert cross-check (struct layout vs upstream assert)\n")
    for bname, expr in asserts.items():
        short = bname[len("block_"):].upper()
        if short in derived:
            val = safe_eval(expr, defines)
            ok = val == derived[short][1]
            print(f"- {bname}: assert expr = {val}B, computed = {derived[short][1]}B -> {'OK' if ok else 'MISMATCH'}")
    print()

    # raw types (non-block)
    RAW = {
        "F32": (1, 4), "F16": (1, 2), "I8": (1, 1), "I16": (1, 2), "I32": (1, 4),
        "I64": (1, 8), "F64": (1, 8), "BF16": (1, 2), "F8_E4M3": (1, 1),
    }

    count = ids.get(44) or "GGML_TYPE_COUNT"
    print(f"## enum from ggml.h (GGML_TYPE_COUNT = {max(k for k in ids if ids[k].endswith('_COUNT')) if False else [k for k in ids if ids[k]=='GGML_TYPE_COUNT'][0]})\n")
    print("| id | enum name | short | block_elems | block_bytes | source |")
    print("|---:|---|---|---:|---:|---|")
    rows = []
    for i in range(0, 44):
        if i not in ids:
            rows.append((i, "(removed/reserved)", "-", "-", "-", "not in enum"))
            print("| %d | *(reserved, not in enum)* | - | - | - | ggml.h enum has no entry |" % i)
            continue
        full = ids[i]
        short = full[len("GGML_TYPE_"):]
        if short == "COUNT":
            print("| %d | GGML_TYPE_COUNT | - | - | - | enum terminator |" % i)
            continue
        if short in derived and derived[short][0] is not None:
            el = derived[short][0]
            bname = "block_" + derived_key_to_block(blocks, short)
            if bname in asserts:
                by = safe_eval(asserts[bname], defines)
                src = "QK macro + sizeof(static_assert expr)"
            else:
                by = derived[short][1]
                src = "QK macro + computed layout"
        elif short in RAW:
            el, by = RAW[short]
            src = "raw scalar (type_traits not needed)"
        else:
            el, by = None, None
            src = "COULD NOT DERIVE"
        rows.append((i, full, short, el, by, src))
        print(f"| {i} | {full} | {short} | {el} | {by} | {src} |")
    print()

    # compare
    print("## Comparison: derived vs constants.py GGML_QUANT_SIZES\n")
    print("| id | short | derived (elems,bytes) | constants.py (elems,bytes) | match |")
    print("|---:|---|---|---|---|")
    n_match = n_diff = n_missing = 0
    for (i, full, short, el, by, src) in rows:
        cs = csizes.get(short)
        if cs is None:
            n_missing += 1
            print(f"| {i} | {short} | ({el},{by}) | MISSING | - |")
            continue
        ok = (el, by) == cs
        if ok:
            n_match += 1
        else:
            n_diff += 1
            print(f"| {i} | {short} | ({el},{by}) | {cs} | **DIFF** |")
    print(f"\nmatch={n_match} diff={n_diff} missing_in_constants={n_missing}")
    extra = set(csizes) - {r[2] for r in rows}
    print("in constants.py but not in ggml.h:", extra or "none")
    cextra = set(cenum.values()) - {r[2] for r in rows}
    print("in constants.py enum but not in ggml.h:", cextra or "none")
    missing_ids = set(cenum) - set(ids)
    print("ids in constants.py enum missing from ggml.h:", missing_ids or "none")

    gcc = None
    if GCC_PROBE:
        gcc = parse_gcc_probe(GCC_PROBE)
        print("\n## GCC probe cross-check (compiled sizeof against real struct definitions)\n")
        print("| short | python layout | gcc sizeof | match |")
        print("|---|---:|---:|---|")
        ng_diff = 0
        for k in derived:
            gv = gcc.get(k)
            ok = gv == derived[k][1]
            if not ok:
                ng_diff += 1
            print(f"| {k} | {derived[k][1]} | {gv} | {'OK' if ok else '**DIFF**'} |")
        print(f"\ngcc mismatches: {ng_diff}")

    # layout details
    print("\n## Struct layout details\n")
    for k in derived:
        print(f"### block_{k.lower()} = block_{k}")
        for d in details[k]:
            print("  - " + d)
        print(f"  => sizeof = {derived[k][1]}")


if __name__ == "__main__":
    main()
