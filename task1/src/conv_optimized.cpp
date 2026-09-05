// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    
    const int p = K / 2;
    const int in_stride = W + (2 * p);
    int tile_size = 32;

    for (int Ty = 0; Ty < H; Ty += tile_size) {
        int tile_end_y = Ty + tile_size < H ? Ty + tile_size : H;
        for (int Tx = 0; Tx < W; Tx += tile_size) {
            int tile_end_x = Tx + tile_size < W ? Tx + tile_size : W;

            for (int oy = Ty; oy < tile_end_y; ++oy) {
                //OPTION 1: TILE SIZE 32 AND USING SIMD FOR 32 BITS (RESULTING IN 4 IN UNROLLS)
                for (int ox = Tx; ox < tile_end_x; ox+=tile_size) {
                    __m256 acc[4] = {_mm256_setzero_ps()};
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            __m256 in_simd[4] = {_mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx))),
                                                 _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 8))),
                                                 _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 16))),
                                                 _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 24)))};
                            __m256 ker_simd = _mm256_set1_ps(ker[ky * K + kx]);
                            acc[0] = _mm256_fmadd_ps(in_simd[0], ker_simd, acc[0]);
                            acc[1] = _mm256_fmadd_ps(in_simd[1], ker_simd, acc[1]);
                            acc[2] = _mm256_fmadd_ps(in_simd[2], ker_simd, acc[2]);
                            acc[3] = _mm256_fmadd_ps(in_simd[3], ker_simd, acc[3]);
                        }
                    }
                    _mm256_store_ps(out + (oy * W + ox + 0), acc[0]);
                    _mm256_store_ps(out + (oy * W + ox + 8), acc[1]);
                    _mm256_store_ps(out + (oy * W + ox + 16), acc[2]);
                    _mm256_store_ps(out + (oy * W + ox + 24), acc[3]);
                }

                // //OPTION 2: TILE SIZE 32 AND USING SIMD FOR 64 BITS (RESULTING IN 2 IN UNROLLS)
                // for (int ox = Tx; ox < tile_end_x; ox+=tile_size) {
                //     __m512 acc[2] = {_mm512_setzero_ps()};
                //     for (int ky = 0; ky < K; ++ky) {
                //         for (int kx = 0; kx < K; ++kx) {
                //             __m512 in_simd[2] = {_mm512_load_ps(in + ((oy + ky) * in_stride + (ox + kx))),
                //                                  _mm512_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 16)))};
                //             __m512 ker_simd = _mm512_set1_ps(ker[ky * K + kx]);
                //             acc[0] = _mm512_fmadd_ps(in_simd[0], ker_simd, acc[0]);
                //             acc[1] = _mm512_fmadd_ps(in_simd[1], ker_simd, acc[1]);
                //         }
                //     }
                //     _mm512_store_ps(out + (oy * W + ox + 0), acc[0]);
                //     _mm512_store_ps(out + (oy * W + ox + 16), acc[1]);
                // }

                // THESE TWO PARTS WERE DONE AS FUN, BUT THEY ARE SLOWER COMPARED TO WHEN THE TILE SIZE WAS 32
                //OPTION 3: TILE SIZE 64 AND USING SIMD FOR 32 BITS (RESULTING IN 8 IN UNROLLS)
                // for (int ox = Tx; ox < tile_end_x; ox+=tile_size) {
                //     __m256 acc[8] = {_mm256_setzero_ps()};
                //     for (int ky = 0; ky < K; ++ky) {
                //         for (int kx = 0; kx < K; ++kx) {
                //             __m256 in_simd[8] = {_mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx))),
                //                                  _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 8))),
                //                                  _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 16))),
                //                                  _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 24))),
                //                                  _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 32))),
                //                                  _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 40))),
                //                                  _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 48))),
                //                                  _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 56)))};
                //             __m256 ker_simd = _mm256_set1_ps(ker[ky * K + kx]);
                //             acc[0] = _mm256_fmadd_ps(in_simd[0], ker_simd, acc[0]);
                //             acc[1] = _mm256_fmadd_ps(in_simd[1], ker_simd, acc[1]);
                //             acc[2] = _mm256_fmadd_ps(in_simd[2], ker_simd, acc[2]);
                //             acc[3] = _mm256_fmadd_ps(in_simd[3], ker_simd, acc[3]);
                //             acc[4] = _mm256_fmadd_ps(in_simd[4], ker_simd, acc[4]);
                //             acc[5] = _mm256_fmadd_ps(in_simd[5], ker_simd, acc[5]);
                //             acc[6] = _mm256_fmadd_ps(in_simd[6], ker_simd, acc[6]);
                //             acc[7] = _mm256_fmadd_ps(in_simd[7], ker_simd, acc[7]);
                //         }
                //     }
                //     _mm256_store_ps(out + (oy * W + ox + 0), acc[0]);
                //     _mm256_store_ps(out + (oy * W + ox + 8), acc[1]);
                //     _mm256_store_ps(out + (oy * W + ox + 16), acc[2]);
                //     _mm256_store_ps(out + (oy * W + ox + 24), acc[3]);
                //     _mm256_store_ps(out + (oy * W + ox + 32), acc[4]);
                //     _mm256_store_ps(out + (oy * W + ox + 40), acc[5]);
                //     _mm256_store_ps(out + (oy * W + ox + 48), acc[6]);
                //     _mm256_store_ps(out + (oy * W + ox + 56), acc[7]);
                // }

                // //OPTION 4: TILE SIZE 64 AND USING SIMD FOR 64 BITS (RESULTING IN 4 IN UNROLLS)
                // for (int ox = Tx; ox < tile_end_x; ox+=tile_size) {
                //     __m512 acc[4] = {_mm512_setzero_ps()};
                //     for (int ky = 0; ky < K; ++ky) {
                //         for (int kx = 0; kx < K; ++kx) {
                //             __m512 in_simd[4] = {_mm512_load_ps(in + ((oy + ky) * in_stride + (ox + kx))), 
                //                                  _mm512_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 16))),
                //                                  _mm512_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 32))),
                //                                  _mm512_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 48)))};
                //             __m512 ker_simd = _mm512_set1_ps(ker[ky * K + kx]);
                //             acc[0] = _mm512_fmadd_ps(in_simd[0], ker_simd, acc[0]);
                //             acc[1] = _mm512_fmadd_ps(in_simd[1], ker_simd, acc[1]);
                //             acc[2] = _mm512_fmadd_ps(in_simd[2], ker_simd, acc[2]);
                //             acc[3] = _mm512_fmadd_ps(in_simd[3], ker_simd, acc[3]);
                //         }
                //     }
                //     _mm512_store_ps(out + (oy * W + ox + 0), acc[0]);
                //     _mm512_store_ps(out + (oy * W + ox + 16), acc[1]);
                //     _mm512_store_ps(out + (oy * W + ox + 32), acc[2]);
                //     _mm512_store_ps(out + (oy * W + ox + 48), acc[3]);
                // }
            }
        }
    }
}
