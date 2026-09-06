// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    
    const int p = K / 2;
    const int in_stride = W + (2 * p);
    int tile_size = 64;

    for (int Ty = 0; Ty < H; Ty += tile_size) {
        int tile_end_y = Ty + tile_size < H ? Ty + tile_size : H;
        for (int Tx = 0; Tx < W; Tx += tile_size) {
            int tile_end_x = Tx + tile_size < W ? Tx + tile_size : W;

            for (int oy = Ty; oy < tile_end_y; ++oy) {
                // ====== TILE SIZE 16 ======//
                // //OPTION 1: TILE SIZE 16 AND USING SIMD FOR 128 BITS (RESULTING IN 4 UNROLLS)
                // for (int ox = Tx; ox < tile_end_x; ox+=tile_size) {
                //     __m128 acc[4] = {_mm_setzero_ps()};
                //     for (int ky = 0; ky < K; ++ky) {
                //         for (int kx = 0; kx < K; ++kx) {
                //             __m128 in_simd[4] = {_mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 4))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 8))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 12)))};
                //             __m128 ker_simd = _mm_set1_ps(ker[ky * K + kx]);
                //             acc[0] = _mm_fmadd_ps(in_simd[0], ker_simd, acc[0]);
                //             acc[1] = _mm_fmadd_ps(in_simd[1], ker_simd, acc[1]);
                //             acc[2] = _mm_fmadd_ps(in_simd[2], ker_simd, acc[2]);
                //             acc[3] = _mm_fmadd_ps(in_simd[3], ker_simd, acc[3]);
                //         }
                //     }
                //     _mm_store_ps(out + (oy * W + ox + 0), acc[0]);
                //     _mm_store_ps(out + (oy * W + ox + 4), acc[1]);
                //     _mm_store_ps(out + (oy * W + ox + 8), acc[2]);
                //     _mm_store_ps(out + (oy * W + ox + 12), acc[3]);
                // }

                // //OPTION 2: TILE SIZE 16 AND USING SIMD FOR 256 BITS (RESULTING IN 2 UNROLLS)
                // for (int ox = Tx; ox < tile_end_x; ox+=tile_size) {
                //     __m256 acc[2] = {_mm256_setzero_ps()};
                //     for (int ky = 0; ky < K; ++ky) {
                //         for (int kx = 0; kx < K; ++kx) {
                //             __m256 in_simd[2] = {_mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx))),
                //                                  _mm256_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 8)))};
                //             __m256 ker_simd = _mm256_set1_ps(ker[ky * K + kx]);
                //             acc[0] = _mm256_fmadd_ps(in_simd[0], ker_simd, acc[0]);
                //             acc[1] = _mm256_fmadd_ps(in_simd[1], ker_simd, acc[1]);
                //         }
                //     }
                //     _mm256_store_ps(out + (oy * W + ox + 0), acc[0]);
                //     _mm256_store_ps(out + (oy * W + ox + 8), acc[1]);
                // }

                // //OPTION 3: TILE SIZE 16 AND USING SIMD FOR 512 BITS (RESULTING IN 1 UNROLLS)
                // for (int ox = Tx; ox < tile_end_x; ox+=tile_size) {
                //     __m512 acc = _mm512_setzero_ps();
                //     for (int ky = 0; ky < K; ++ky) {
                //         for (int kx = 0; kx < K; ++kx) {
                //             __m512 in_simd = _mm512_load_ps(in + ((oy + ky) * in_stride + (ox + kx)));
                //             __m512 ker_simd = _mm512_set1_ps(ker[ky * K + kx]);
                //             acc = _mm512_fmadd_ps(in_simd, ker_simd, acc);
                //         }
                //     }
                //     _mm512_store_ps(out + (oy * W + ox + 0), acc);
                // }

                // ====== TILE SIZE 32 ======//
                // //OPTION 1: TILE SIZE 32 AND USING SIMD FOR 128 BITS (RESULTING IN 8 IN UNROLLS)
                // for (int ox = Tx; ox < tile_end_x; ox+=tile_size) {
                //     __m128 acc[8] = {_mm_setzero_ps()};
                //     for (int ky = 0; ky < K; ++ky) {
                //         for (int kx = 0; kx < K; ++kx) {
                //             __m128 in_simd[8] = {_mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 4))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 8))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 12))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 16))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 20))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 24))),
                //                                  _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 28)))};
                //             __m128 ker_simd = _mm_set1_ps(ker[ky * K + kx]);
                //             acc[0] = _mm_fmadd_ps(in_simd[0], ker_simd, acc[0]);
                //             acc[1] = _mm_fmadd_ps(in_simd[1], ker_simd, acc[1]);
                //             acc[2] = _mm_fmadd_ps(in_simd[2], ker_simd, acc[2]);
                //             acc[3] = _mm_fmadd_ps(in_simd[3], ker_simd, acc[3]);
                //             acc[4] = _mm_fmadd_ps(in_simd[4], ker_simd, acc[4]);
                //             acc[5] = _mm_fmadd_ps(in_simd[5], ker_simd, acc[5]);
                //             acc[6] = _mm_fmadd_ps(in_simd[6], ker_simd, acc[6]);
                //             acc[7] = _mm_fmadd_ps(in_simd[7], ker_simd, acc[7]);
                //         }
                //     }
                //     _mm_store_ps(out + (oy * W + ox + 0), acc[0]);
                //     _mm_store_ps(out + (oy * W + ox + 4), acc[1]);
                //     _mm_store_ps(out + (oy * W + ox + 8), acc[2]);
                //     _mm_store_ps(out + (oy * W + ox + 12), acc[3]);
                //     _mm_store_ps(out + (oy * W + ox + 16), acc[4]);
                //     _mm_store_ps(out + (oy * W + ox + 20), acc[5]);
                //     _mm_store_ps(out + (oy * W + ox + 24), acc[6]);
                //     _mm_store_ps(out + (oy * W + ox + 28), acc[7]);
                // }

                //OPTION 2: TILE SIZE 32 AND USING SIMD FOR 256 BITS (RESULTING IN 4 IN UNROLLS)
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

                //OPTION 3: TILE SIZE 32 AND USING SIMD FOR 512 BITS (RESULTING IN 2 IN UNROLLS)
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

                // ====== TILE SIZE 64 ======//
                // THESE TWO PARTS WERE DONE AS FUN, BUT THEY ARE SLOWER COMPARED TO WHEN THE TILE SIZE WAS 32
                // //OPTION 1: TILE SIZE 64 AND USING SIMD FOR 128 BITS (RESULTING IN 16 IN UNROLLS)
                // for (int ox = Tx; ox < tile_end_x; ox+=tile_size) {
                //     __m128 acc[16] = {_mm_setzero_ps()};
                //     for (int ky = 0; ky < K; ++ky) {
                //         for (int kx = 0; kx < K; ++kx) {
                //             __m128 in_simd[16] = {_mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 4))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 8))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 12))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 16))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 20))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 24))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 28))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 32))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 36))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 40))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 44))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 48))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 52))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 56))),
                //                                   _mm_load_ps(in + ((oy + ky) * in_stride + (ox + kx + 60)))};
                //             __m128 ker_simd = _mm_set1_ps(ker[ky * K + kx]);
                //             acc[0] = _mm_fmadd_ps(in_simd[0], ker_simd, acc[0]);
                //             acc[1] = _mm_fmadd_ps(in_simd[1], ker_simd, acc[1]);
                //             acc[2] = _mm_fmadd_ps(in_simd[2], ker_simd, acc[2]);
                //             acc[3] = _mm_fmadd_ps(in_simd[3], ker_simd, acc[3]);
                //             acc[4] = _mm_fmadd_ps(in_simd[4], ker_simd, acc[4]);
                //             acc[5] = _mm_fmadd_ps(in_simd[5], ker_simd, acc[5]);
                //             acc[6] = _mm_fmadd_ps(in_simd[6], ker_simd, acc[6]);
                //             acc[7] = _mm_fmadd_ps(in_simd[7], ker_simd, acc[7]);
                //             acc[8] = _mm_fmadd_ps(in_simd[8], ker_simd, acc[8]);
                //             acc[9] = _mm_fmadd_ps(in_simd[9], ker_simd, acc[9]);
                //             acc[10] = _mm_fmadd_ps(in_simd[10], ker_simd, acc[10]);
                //             acc[11] = _mm_fmadd_ps(in_simd[11], ker_simd, acc[11]);
                //             acc[12] = _mm_fmadd_ps(in_simd[12], ker_simd, acc[12]);
                //             acc[13] = _mm_fmadd_ps(in_simd[13], ker_simd, acc[13]);
                //             acc[14] = _mm_fmadd_ps(in_simd[14], ker_simd, acc[14]);
                //             acc[15] = _mm_fmadd_ps(in_simd[15], ker_simd, acc[15]);
                //         }
                //     }
                //     _mm_store_ps(out + (oy * W + ox + 0), acc[0]);
                //     _mm_store_ps(out + (oy * W + ox + 4), acc[1]);
                //     _mm_store_ps(out + (oy * W + ox + 8), acc[2]);
                //     _mm_store_ps(out + (oy * W + ox + 12), acc[3]);
                //     _mm_store_ps(out + (oy * W + ox + 16), acc[4]);
                //     _mm_store_ps(out + (oy * W + ox + 20), acc[5]);
                //     _mm_store_ps(out + (oy * W + ox + 24), acc[6]);
                //     _mm_store_ps(out + (oy * W + ox + 28), acc[7]);
                //     _mm_store_ps(out + (oy * W + ox + 32), acc[8]);
                //     _mm_store_ps(out + (oy * W + ox + 36), acc[9]);
                //     _mm_store_ps(out + (oy * W + ox + 40), acc[10]);
                //     _mm_store_ps(out + (oy * W + ox + 44), acc[11]);
                //     _mm_store_ps(out + (oy * W + ox + 48), acc[12]);
                //     _mm_store_ps(out + (oy * W + ox + 52), acc[13]);
                //     _mm_store_ps(out + (oy * W + ox + 56), acc[14]);
                //     _mm_store_ps(out + (oy * W + ox + 60), acc[15]);
                // }

                // //OPTION 2: TILE SIZE 64 AND USING SIMD FOR 256 BITS (RESULTING IN 8 IN UNROLLS)
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

                // //OPTION 3: TILE SIZE 64 AND USING SIMD FOR 512 BITS (RESULTING IN 4 IN UNROLLS)
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
