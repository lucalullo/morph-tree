/* MorphTree-owned prototype: ordered five-plane histogram, ABI v1.
 * No scoring policy, split rule, sampling, tree growth or learner is compiled
 * into this primitive. Caller supplies stable owned buffers; output never aliases
 * an input. No fast-math, reassociation or reduction reordering is permitted.
 * SOURCE ONLY until a real compiler/build/test result is recorded.
 */
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <string.h>

_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && FLT_RADIX == 2,
               "MorphTree histogram ABI requires binary64 doubles");
_Static_assert(sizeof(size_t) == 8, "Prototype currently qualified for 64-bit ABI only");

#if defined(_WIN32)
#define MT_EXPORT __declspec(dllexport)
#else
#define MT_EXPORT __attribute__((visibility("default")))
#endif

enum { MT_OK=0, MT_BAD_ARGUMENT=1, MT_BAD_INDEX=2, MT_BAD_BIN=3, MT_OVERFLOW=4 };

MT_EXPORT uint32_t mt_histogram_abi_version(void) { return 1u; }

static uint32_t read_bin(const void *bins, uint32_t width, uint64_t row) {
    /* memcpy also avoids alignment/aliasing undefined behavior at the ABI edge. */
    if (width == 1u) return ((const uint8_t *)bins)[row];
    if (width == 2u) { uint16_t v; memcpy(&v, (const uint8_t *)bins + row*2u, 2u); return v; }
    { uint32_t v; memcpy(&v, (const uint8_t *)bins + row*4u, 4u); return v; }
}

MT_EXPORT int mt_histogram_f64_v1(
        const void *bins, uint32_t width, uint64_t rows,
        const uint64_t *ids, uint64_t selected,
        const double *g, const double *h, const double *r, const double *u,
        uint64_t bin_count, double *out, uint64_t output_doubles) {
    if ((width!=1u && width!=2u && width!=4u) || bin_count==0u || !out)
        return MT_BAD_ARGUMENT;
    if (bin_count > SIZE_MAX/(5u*sizeof(double)) || rows > SIZE_MAX/width)
        return MT_OVERFLOW;
    if (output_doubles < 5u*bin_count) return MT_BAD_ARGUMENT;
    if (selected && (!bins || !ids || !g || !h || !r || !u)) return MT_BAD_ARGUMENT;
    /* Validate first: ordinary invalid requests do not alter the output.
     * The Python prototype retains the GIL and buffer exports throughout. */
    for (uint64_t k=0; k<selected; ++k) {
        const uint64_t i=ids[k];
        if (i>=rows) return MT_BAD_INDEX;
        if (read_bin(bins,width,i)>=bin_count) return MT_BAD_BIN;
    }
    memset(out,0,(size_t)(5u*bin_count*sizeof(double)));
    double *counts=out, *gs=out+bin_count, *hs=out+2u*bin_count;
    double *rs=out+3u*bin_count, *us=out+4u*bin_count;
    for (uint64_t k=0; k<selected; ++k) {
        const uint64_t i=ids[k];
        const uint32_t b=read_bin(bins,width,i);
        counts[b]+=1.0;
        gs[b]+=g[i]; hs[b]+=h[i]; rs[b]+=r[i]; us[b]+=u[i];
    }
    return MT_OK;
}
