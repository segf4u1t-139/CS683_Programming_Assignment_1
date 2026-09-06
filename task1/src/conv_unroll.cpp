// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {

        // IMPLEMENTATION FOR UNROLL OF 8 INDEPENDENT ADDITIONS
        for (int ox = 0; ox < W; ox+=8) {
            float acc[8] = {0.0f};
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc[0] += in[(oy + ky) * in_stride + (ox + kx + 0)] * ker[ky * K + kx];
                    acc[1] += in[(oy + ky) * in_stride + (ox + kx + 1)] * ker[ky * K + kx];
                    acc[2] += in[(oy + ky) * in_stride + (ox + kx + 2)] * ker[ky * K + kx];
                    acc[3] += in[(oy + ky) * in_stride + (ox + kx + 3)] * ker[ky * K + kx];
                    acc[4] += in[(oy + ky) * in_stride + (ox + kx + 4)] * ker[ky * K + kx];
                    acc[5] += in[(oy + ky) * in_stride + (ox + kx + 5)] * ker[ky * K + kx];
                    acc[6] += in[(oy + ky) * in_stride + (ox + kx + 6)] * ker[ky * K + kx];
                    acc[7] += in[(oy + ky) * in_stride + (ox + kx + 7)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox + 0] = acc[0];
            out[oy * W + ox + 1] = acc[1];
            out[oy * W + ox + 2] = acc[2];
            out[oy * W + ox + 3] = acc[3];
            out[oy * W + ox + 4] = acc[4];
            out[oy * W + ox + 5] = acc[5];
            out[oy * W + ox + 6] = acc[6];
            out[oy * W + ox + 7] = acc[7];
        }

        // // IMPLEMENTATION FOR UNROLL OF 16 INDEPENDENT ADDITIONS
        // for (int ox = 0; ox < W; ox+=16) {
        //     float acc[16] = {0.0f};
        //     for (int ky = 0; ky < K; ++ky) {
        //         for (int kx = 0; kx < K; ++kx) {
        //             acc[0] += in[(oy + ky) * in_stride + (ox + kx + 0)] * ker[ky * K + kx];
        //             acc[1] += in[(oy + ky) * in_stride + (ox + kx + 1)] * ker[ky * K + kx];
        //             acc[2] += in[(oy + ky) * in_stride + (ox + kx + 2)] * ker[ky * K + kx];
        //             acc[3] += in[(oy + ky) * in_stride + (ox + kx + 3)] * ker[ky * K + kx];
        //             acc[4] += in[(oy + ky) * in_stride + (ox + kx + 4)] * ker[ky * K + kx];
        //             acc[5] += in[(oy + ky) * in_stride + (ox + kx + 5)] * ker[ky * K + kx];
        //             acc[6] += in[(oy + ky) * in_stride + (ox + kx + 6)] * ker[ky * K + kx];
        //             acc[7] += in[(oy + ky) * in_stride + (ox + kx + 7)] * ker[ky * K + kx];
        //             acc[8] += in[(oy + ky) * in_stride + (ox + kx + 8)] * ker[ky * K + kx];
        //             acc[9] += in[(oy + ky) * in_stride + (ox + kx + 9)] * ker[ky * K + kx];
        //             acc[10] += in[(oy + ky) * in_stride + (ox + kx + 10)] * ker[ky * K + kx];
        //             acc[11] += in[(oy + ky) * in_stride + (ox + kx + 11)] * ker[ky * K + kx];
        //             acc[12] += in[(oy + ky) * in_stride + (ox + kx + 12)] * ker[ky * K + kx];
        //             acc[13] += in[(oy + ky) * in_stride + (ox + kx + 13)] * ker[ky * K + kx];
        //             acc[14] += in[(oy + ky) * in_stride + (ox + kx + 14)] * ker[ky * K + kx];
        //             acc[15] += in[(oy + ky) * in_stride + (ox + kx + 15)] * ker[ky * K + kx];
        //         }
        //     }
        //     out[oy * W + ox + 0] = acc[0];
        //     out[oy * W + ox + 1] = acc[1];
        //     out[oy * W + ox + 2] = acc[2];
        //     out[oy * W + ox + 3] = acc[3];
        //     out[oy * W + ox + 4] = acc[4];
        //     out[oy * W + ox + 5] = acc[5];
        //     out[oy * W + ox + 6] = acc[6];
        //     out[oy * W + ox + 7] = acc[7];
        //     out[oy * W + ox + 8] = acc[8];
        //     out[oy * W + ox + 9] = acc[9];
        //     out[oy * W + ox + 10] = acc[10];
        //     out[oy * W + ox + 11] = acc[11];
        //     out[oy * W + ox + 12] = acc[12];
        //     out[oy * W + ox + 13] = acc[13];
        //     out[oy * W + ox + 14] = acc[14];
        //     out[oy * W + ox + 15] = acc[15];
        // }
    }
}
