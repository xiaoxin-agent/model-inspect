/*
 * Third-source probe: compile the REAL struct definitions from upstream/ggml-common.h
 * and print sizeof and _Alignof for every block_* type.
 *
 * Usage (from internal/gguf/docs):
 *   gcc -std=c11 -o /tmp/probe probe_sizes.c -I. && /tmp/probe > /tmp/probe.txt
 *   python3 derive_types.py --gcc-probe /tmp/probe.txt
 *
 * Why GGML_COMMON_DECL_C: it defines ggml_half = uint16_t and ggml_half2 = uint32_t,
 * i.e. the CPU-side declarations used by ggml.c, and they are the same sizes as the
 * 2-element half vectors used by the GPU backends. The probe only measures
 * sizeof/_Alignof; it runs no ggml code.
 */
#include <stdio.h>
#include <stdint.h>
#define GGML_COMMON_DECL_C
#include "upstream/ggml-common.h"
#define P(t) printf("%-16s %zu %zu\n", #t, sizeof(t), _Alignof(t))
int main(void) {
    P(block_q4_0);  P(block_q4_1);  P(block_q5_0);  P(block_q5_1);
    P(block_q8_0);  P(block_q8_1);  P(block_q2_K);  P(block_q3_K);
    P(block_q4_K);  P(block_q5_K);  P(block_q6_K);  P(block_q8_K);
    P(block_q1_0);  P(block_q2_0);  P(block_mxfp4); P(block_nvfp4);
    P(block_tq1_0); P(block_tq2_0); P(block_iq2_xxs); P(block_iq2_xs);
    P(block_iq2_s); P(block_iq3_xxs); P(block_iq3_s); P(block_iq1_s);
    P(block_iq1_m); P(block_iq4_nl); P(block_iq4_xs);
    return 0;
}
