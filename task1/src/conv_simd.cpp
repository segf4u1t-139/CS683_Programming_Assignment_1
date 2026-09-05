// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {

        // THE NEEDED IMPLEMENTATION FOR SIMD FOR 128 BITS (16 BYTES => 4 FLOAT VALUES WHICH WE KNOW CAN PARALLELIZE).
        for (int ox = 0; ox < W; ox+=4) {
            __m128 acc = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m128 in_simd = _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx)));
                    __m128 ker_simd = _mm_set1_ps(ker[ky * K + kx]);
                    acc = _mm_fmadd_ps(in_simd, ker_simd, acc);
                }
            }
            _mm_store_ps(out + (oy * W + ox), acc);
        }

        // // THE NEEDED IMPLEMENTATION FOR SIMD FOR 256 BITS (32 BYTES => 8 FLOAT VALUES WHICH WE KNOW CAN PARALLELIZE).
        // for (int ox = 0; ox < W; ox+=8) {
        //     __m256 acc = _mm256_setzero_ps();
        //     for (int ky = 0; ky < K; ++ky) {
        //         for (int kx = 0; kx < K; ++kx) {
        //             __m256 in_simd = _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx)));
        //             __m256 ker_simd = _mm256_set1_ps(ker[ky * K + kx]);
        //             acc = _mm256_fmadd_ps(in_simd, ker_simd, acc);
        //         }
        //     }
        //     _mm256_store_ps(out + (oy * W + ox), acc);
        // }

        // THE NEEDED IMPLEMENTATION FOR SIMD FOR 512 BITS (64 BYTES => 16 FLOAT VALUES WHICH MIGHT FAIL FOR W NOT DIVISIBLE BY 16).
        // for (int ox = 0; ox < W; ox+=16) {
        //     __m512 acc = _mm512_setzero_ps();
        //     for (int ky = 0; ky < K; ++ky) {
        //         for (int kx = 0; kx < K; ++kx) {
        //             __m512 in_simd = _mm512_load_ps(in + ((oy + ky) * in_stride + (ox + kx)));
        //             __m512 ker_simd = _mm512_set1_ps(ker[ky * K + kx]);
        //             acc = _mm512_fmadd_ps(in_simd, ker_simd, acc);
        //         }
        //     }
        //     _mm512_store_ps(out + (oy * W + ox), acc);
        // }
    }
}
