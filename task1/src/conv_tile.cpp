// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    int tile_size = 1024;

    for (int Ty = 0; Ty < H; Ty += tile_size) {
        int tile_end_y = Ty + tile_size < H ? Ty + tile_size : H;
        for (int Tx = 0; Tx < W; Tx += tile_size) {
            int tile_end_x = Tx + tile_size < W ? Tx + tile_size : W;

            for (int oy = Ty; oy < tile_end_y; ++oy) {
                for (int ox = Tx; ox < tile_end_x; ++ox) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}
