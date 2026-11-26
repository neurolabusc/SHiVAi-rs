#include <stddef.h>
#if defined(__clang__) && __has_include(<arm_sme.h>) // GCC has the header but not these attributes
#include <arm_sme.h>
#include <sys/sysctl.h>

__attribute__((target("sme"))) __arm_locally_streaming
static int width(void) { return svcntsw(); }

int shivai_sme_available(void) {
    int enabled = 0;
    size_t size = sizeof(enabled);
    return !sysctlbyname("hw.optional.arm.SME_F32F32", &enabled, &size, NULL, 0)
        && enabled && width() == 16;
}

// Four 16x16 accumulators share each weight vector across 64 spatial positions.
__attribute__((target("sme"))) __arm_locally_streaming __arm_new("za")
void shivai_sme_matmul(const float *a, const float *b, float *out,
                       size_t rows, size_t cols, size_t depth) {
    svbool_t all = svptrue_b32();
    for (size_t col = 0; col < cols; col += 16) {
        svbool_t tail = svwhilelt_b32((uint64_t)col, (uint64_t)cols);
        svzero_za();
        for (size_t k = 0; k < depth; k++) {
            svfloat32_t w = svld1_f32(tail, b + k * cols + col);
            const float *x = a + k * 64;
            svmopa_za32_f32_m(0, all, tail, svld1_f32(all, x), w);
            svmopa_za32_f32_m(1, all, tail, svld1_f32(all, x + 16), w);
            svmopa_za32_f32_m(2, all, tail, svld1_f32(all, x + 32), w);
            svmopa_za32_f32_m(3, all, tail, svld1_f32(all, x + 48), w);
        }
        for (size_t r = 0; r < rows && r < 16; r++)
            svst1_hor_za32(0, r, tail, out + r * cols + col);
        for (size_t r = 16; r < rows && r < 32; r++)
            svst1_hor_za32(1, r - 16, tail, out + r * cols + col);
        for (size_t r = 32; r < rows && r < 48; r++)
            svst1_hor_za32(2, r - 32, tail, out + r * cols + col);
        for (size_t r = 48; r < rows; r++)
            svst1_hor_za32(3, r - 48, tail, out + r * cols + col);
    }
}
#else
int shivai_sme_available(void) { return 0; }
void shivai_sme_matmul(const float *a, const float *b, float *out,
                       size_t rows, size_t cols, size_t depth) {}
#endif
