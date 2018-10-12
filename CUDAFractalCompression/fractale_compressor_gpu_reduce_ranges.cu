#undef NDEBUG

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <inttypes.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "ifs_transform_data.hpp"

#include "cuda_call_checker.cuh"
#include "affine_transformer_gpu.cuh"
#include "fractale_compressor_gpu_reduce_ranges.cuh"

__constant__ uint8_t dc_affine_transform_size4[7][16];
__constant__ uint8_t dc_affine_transform_size8[7][64];
__constant__ uint8_t dc_affine_transform_size16[7][256];

void init_fcrr_affine_transformer(int size) {

	uint8_t h_affine_transform_size4[7][16] = {
		//90
		{ 12,  8,  4,  0,
		13,  9,  5,  1,
		14, 10,  6,  2,
		15, 11,  7,  3 },
		//180
		{ 15, 14, 13, 12,
		11, 10,  9,  8,
		7,  6,  5,  4,
		3,  2,  1,  0 },
		//270
		{ 3,  7, 11, 15,
		2,  6, 10, 14,
		1,  5,  9, 13,
		0,  4,  8, 12 },
		//mirror 0
		{ 3,  2,  1,  0,
		7,  6,  5,  4,
		11, 10,  9,  8,
		15, 14, 13, 12 },
		//mirror 90
		{ 15, 11,  7,  3,
		14, 10,  6,  2,
		13,  9,  5,  1,
		12,  8,  4,  0 },
		//mirror 180
		{ 12, 13, 14, 15,
		8,  9, 10, 11,
		4,  5,  6,  7,
		0,  1,  2,  3 },
		//mirror 270
		{ 0,  4,  8, 12,
		1,  5,  9, 13,
		2,  6, 10, 14,
		3,  7, 11, 15 }
	};

	uint8_t h_affine_transform_size8[7][64] = {
		//90
		{ 56, 48, 40, 32, 24, 16,  8,  0,
		57, 49, 41, 33, 25, 17,  9,  1,
		58, 50, 42, 34, 26, 18, 10,  2,
		59, 51, 43, 35, 27, 19, 11,  3,
		60, 52, 44, 36, 28, 20, 12,  4,
		61, 53, 45, 37, 29, 21, 13,  5,
		62, 54, 46, 38, 30, 22, 14,  6,
		63, 55, 47, 39, 31, 23, 15,  7 },
		//180
		{ 63, 62, 61, 60, 59, 58, 57, 56,
		55, 54, 53, 52, 51, 50, 49, 48,
		47, 46, 45, 44, 43, 42, 41, 40,
		39, 38, 37, 36, 35, 34, 33, 32,
		31, 30, 29, 28, 27, 26, 25, 24,
		23, 22, 21, 20, 19, 18, 17, 16,
		15, 14, 13, 12, 11, 10,  9,  8,
		7,  6,  5,  4,  3,  2,  1,  0 },
		//270
		{ 7, 15, 23, 31, 39, 47, 55, 63,
		6, 14, 22, 30, 38, 46, 54, 62,
		5, 13, 21, 29, 37, 45, 53, 61,
		4, 12, 20, 28, 36, 44, 52, 60,
		3, 11, 19, 27, 35, 43, 51, 59,
		2, 10, 18, 26, 34, 42, 50, 58,
		1,  9, 17, 25, 33, 41, 49, 57,
		0,  8, 16, 24, 32, 40, 48, 56 },
		//mirror
		{ 7,  6,  5,  4,  3,  2,  1,  0,
		15, 14, 13, 12, 11, 10,  9,  8,
		23, 22, 21, 20, 19, 18, 17, 16,
		31, 30, 29, 28, 27, 26, 25, 24,
		39, 38, 37, 36, 35, 34, 33, 32,
		47, 46, 45, 44, 43, 42, 41, 40,
		55, 54, 53, 52, 51, 50, 49, 48,
		63, 62, 61, 60, 59, 58, 57, 56 },
		//mirror 90
		{ 63, 55, 47, 39, 31, 23, 15,  7,
		62, 54, 46, 38, 30, 22, 14,  6,
		61, 53, 45, 37, 29, 21, 13,  5,
		60, 52, 44, 36, 28, 20, 12,  4,
		59, 51, 43, 35, 27, 19, 11,  3,
		58, 50, 42, 34, 26, 18, 10,  2,
		57, 49, 41, 33, 25, 17,  9,  1,
		56, 48, 40, 32, 24, 16,  8,  0 },
		//mirror 180
		{ 56, 57, 58, 59, 60, 61, 62, 63,
		48, 49, 50, 51, 52, 53, 54, 55,
		40, 41, 42, 43, 44, 45, 46, 47,
		32, 33, 34, 35, 36, 37, 38, 39,
		24, 25, 26, 27, 28, 29, 30, 31,
		16, 17, 18, 19, 20, 21, 22, 23,
		8,  9, 10, 11, 12, 13, 14, 15,
		0,  1,  2,  3,  4,  5,  6,  7 },
		//mirror 270
		{ 0,  8, 16, 24, 32, 40, 48, 56,
		1,  9, 17, 25, 33, 41, 49, 57,
		2, 10, 18, 26, 34, 42, 50, 58,
		3, 11, 19, 27, 35, 43, 51, 59,
		4, 12, 20, 28, 36, 44, 52, 60,
		5, 13, 21, 29, 37, 45, 53, 61,
		6, 14, 22, 30, 38, 46, 54, 62,
		7, 15, 23, 31, 39, 47, 55, 63 }
	};

	uint8_t h_affine_transform_size16[7][256] = {
		//90
		{ 240,224,208,192,176,160,144,128,112, 96, 80, 64, 48, 32, 16,  0,
		241,225,209,193,177,161,145,129,113, 97, 81, 65, 49, 33, 17,  1,
		242,226,210,194,178,162,146,130,114, 98, 82, 66, 50, 34, 18,  2,
		243,227,211,195,179,163,147,131,115, 99, 83, 67, 51, 35, 19,  3,
		244,228,212,196,180,164,148,132,116,100, 84, 68, 52, 36, 20,  4,
		245,229,213,197,181,165,149,133,117,101, 85, 69, 53, 37, 21,  5,
		246,230,214,198,182,166,150,134,118,102, 86, 70, 54, 38, 22,  6,
		247,231,215,199,183,167,151,135,119,103, 87, 71, 55, 39, 23,  7,
		248,232,216,200,184,168,152,136,120,104, 88, 72, 56, 40, 24,  8,
		249,233,217,201,185,169,153,137,121,105, 89, 73, 57, 41, 25,  9,
		250,234,218,202,186,170,154,138,122,106, 90, 74, 58, 42, 26, 10,
		251,235,219,203,187,171,155,139,123,107, 91, 75, 59, 43, 27, 11,
		252,236,220,204,188,172,156,140,124,108, 92, 76, 60, 44, 28, 12,
		253,237,221,205,189,173,157,141,125,109, 93, 77, 61, 45, 29, 13,
		254,238,222,206,190,174,158,142,126,110, 94, 78, 62, 46, 30, 14,
		255,239,223,207,191,175,159,143,127,111, 95, 79, 63, 47, 31, 15 },
		//180
		{ 255,254,253,252,251,250,249,248,247,246,245,244,243,242,241,240,
		239,238,237,236,235,234,233,232,231,230,229,228,227,226,225,224,
		223,222,221,220,219,218,217,216,215,214,213,212,211,210,209,208,
		207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,
		191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,
		175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,160,
		159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,144,
		143,142,141,140,139,138,137,136,135,134,133,132,131,130,129,128,
		127,126,125,124,123,122,121,120,119,118,117,116,115,114,113,112,
		111,110,109,108,107,106,105,104,103,102,101,100, 99, 98, 97, 96,
		95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80,
		79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64,
		63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
		47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
		15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0 },
		//270
		{ 15, 31, 47, 63, 79, 95,111,127,143,159,175,191,207,223,239,255,
		14, 30, 46, 62, 78, 94,110,126,142,158,174,190,206,222,238,254,
		13, 29, 45, 61, 77, 93,109,125,141,157,173,189,205,221,237,253,
		12, 28, 44, 60, 76, 92,108,124,140,156,172,188,204,220,236,252,
		11, 27, 43, 59, 75, 91,107,123,139,155,171,187,203,219,235,251,
		10, 26, 42, 58, 74, 90,106,122,138,154,170,186,202,218,234,250,
		9, 25, 41, 57, 73, 89,105,121,137,153,169,185,201,217,233,249,
		8, 24, 40, 56, 72, 88,104,120,136,152,168,184,200,216,232,248,
		7, 23, 39, 55, 71, 87,103,119,135,151,167,183,199,215,231,247,
		6, 22, 38, 54, 70, 86,102,118,134,150,166,182,198,214,230,246,
		5, 21, 37, 53, 69, 85,101,117,133,149,165,181,197,213,229,245,
		4, 20, 36, 52, 68, 84,100,116,132,148,164,180,196,212,228,244,
		3, 19, 35, 51, 67, 83, 99,115,131,147,163,179,195,211,227,243,
		2, 18, 34, 50, 66, 82, 98,114,130,146,162,178,194,210,226,242,
		1, 17, 33, 49, 65, 81, 97,113,129,145,161,177,193,209,225,241,
		0, 16, 32, 48, 64, 80, 96,112,128,144,160,176,192,208,224,240 },
		//mirror
		{ 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
		47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
		63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
		79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64,
		95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80,
		111,110,109,108,107,106,105,104,103,102,101,100, 99, 98, 97, 96,
		127,126,125,124,123,122,121,120,119,118,117,116,115,114,113,112,
		143,142,141,140,139,138,137,136,135,134,133,132,131,130,129,128,
		159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,144,
		175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,160,
		191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,
		207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,
		223,222,221,220,219,218,217,216,215,214,213,212,211,210,209,208,
		239,238,237,236,235,234,233,232,231,230,229,228,227,226,225,224,
		255,254,253,252,251,250,249,248,247,246,245,244,243,242,241,240 },
		//mirror 90
		{ 255,239,223,207,191,175,159,143,127,111, 95, 79, 63, 47, 31, 15,
		254,238,222,206,190,174,158,142,126,110, 94, 78, 62, 46, 30, 14,
		253,237,221,205,189,173,157,141,125,109, 93, 77, 61, 45, 29, 13,
		252,236,220,204,188,172,156,140,124,108, 92, 76, 60, 44, 28, 12,
		251,235,219,203,187,171,155,139,123,107, 91, 75, 59, 43, 27, 11,
		250,234,218,202,186,170,154,138,122,106, 90, 74, 58, 42, 26, 10,
		249,233,217,201,185,169,153,137,121,105, 89, 73, 57, 41, 25,  9,
		248,232,216,200,184,168,152,136,120,104, 88, 72, 56, 40, 24,  8,
		247,231,215,199,183,167,151,135,119,103, 87, 71, 55, 39, 23,  7,
		246,230,214,198,182,166,150,134,118,102, 86, 70, 54, 38, 22,  6,
		245,229,213,197,181,165,149,133,117,101, 85, 69, 53, 37, 21,  5,
		244,228,212,196,180,164,148,132,116,100, 84, 68, 52, 36, 20,  4,
		243,227,211,195,179,163,147,131,115, 99, 83, 67, 51, 35, 19,  3,
		242,226,210,194,178,162,146,130,114, 98, 82, 66, 50, 34, 18,  2,
		241,225,209,193,177,161,145,129,113, 97, 81, 65, 49, 33, 17,  1,
		240,224,208,192,176,160,144,128,112, 96, 80, 64, 48, 32, 16,  0 },
		//mirror 180
		{ 240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,
		224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,
		208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,
		192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,
		176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
		160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,
		144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
		128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,
		112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,
		96 ,97, 98, 99,100,101,102,103,104,105,106,107,108,109,110,111,
		80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
		64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
		48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
		32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
		16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
		0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
		//mirror 270
		{ 0, 16, 32, 48, 64, 80, 96,112,128,144,160,176,192,208,224,240,
		1, 17, 33, 49, 65, 81, 97,113,129,145,161,177,193,209,225,241,
		2, 18, 34, 50, 66, 82, 98,114,130,146,162,178,194,210,226,242,
		3, 19, 35, 51, 67, 83, 99,115,131,147,163,179,195,211,227,243,
		4, 20, 36, 52, 68, 84,100,116,132,148,164,180,196,212,228,244,
		5, 21, 37, 53, 69, 85,101,117,133,149,165,181,197,213,229,245,
		6, 22, 38, 54, 70, 86,102,118,134,150,166,182,198,214,230,246,
		7, 23, 39, 55, 71, 87,103,119,135,151,167,183,199,215,231,247,
		8, 24, 40, 56, 72, 88,104,120,136,152,168,184,200,216,232,248,
		9, 25, 41, 57, 73, 89,105,121,137,153,169,185,201,217,233,249,
		10, 26, 42, 58, 74, 90,106,122,138,154,170,186,202,218,234,250,
		11, 27, 43, 59, 75, 91,107,123,139,155,171,187,203,219,235,251,
		12, 28, 44, 60, 76, 92,108,124,140,156,172,188,204,220,236,252,
		13, 29, 45, 61, 77, 93,109,125,141,157,173,189,205,221,237,253,
		14, 30, 46, 62, 78, 94,110,126,142,158,174,190,206,222,238,254,
		15, 31, 47, 63, 79, 95,111,127,143,159,175,191,207,223,239,255 }
	};

	if (size == 4) {
		CHECK(cudaMemcpyToSymbol(dc_affine_transform_size4, h_affine_transform_size4, sizeof(uint8_t) * 7 * 16));
		std::cout << "size 4 copyed" << std::endl;
	}
	//cudaMemC
	else if (size == 8) {
		CHECK(cudaMemcpyToSymbol(dc_affine_transform_size8, h_affine_transform_size8, sizeof(uint8_t) * 7 * 64));
		std::cout << "size 8 copyed" << std::endl;
	}
	else if (size == 16) {
		CHECK(cudaMemcpyToSymbol(dc_affine_transform_size16, h_affine_transform_size16, sizeof(uint8_t) * 7 * 256));
		std::cout << "size 16 copyed" << std::endl;
	}
	CHECK(cudaDeviceSynchronize());
}

/*
input:
	uint8_t* d_orig_img		: 一般的な画像の形式
output:
	uint8_t* d_ranges	:ブロック化された画像の形式
	uint8_t* d_domains	:ブロック化された画像の形式 画像サイズ1/2 ブロックサイズは同じ

	フラクタル圧縮に必要な画像配列を生成する
	カーネルを呼ぶ時のブロック数でドメイン（圧縮），レンジのブロックを決定する
	fc_make_range_n_domain<<<grid, (block_x, block_y)>>>
	のblock_x, block_yがブロックの大きさになる
*/
__global__ void fcrr_make_domains_n_ranges(uint8_t* d_orig_img, 
									       uint8_t* d_ranges,
									       uint8_t* d_domains){

	uint32_t rdblock_id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y;
	uint32_t rdblock_thread_id = blockDim.x * threadIdx.y + threadIdx.x;
	uint32_t rdblock_array_id = rdblock_id + rdblock_thread_id;

	uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t normal_array_id = y * (gridDim.x * blockDim.x) + x;

	d_ranges[rdblock_array_id] = d_orig_img[normal_array_id];

	if( (blockIdx.x >= (gridDim.x >> 1) ) || ( blockIdx.y >= (gridDim.y >> 1) ) ){
		return;
	}

	uint32_t half_id = (blockIdx.y * (gridDim.x >> 1) + blockIdx.x) * blockDim.x * blockDim.y;

	uint32_t idx1 = d_orig_img[2 * y * (gridDim.x * blockDim.x) + 2 * x];
	uint32_t idx2 = d_orig_img[2 * y * (gridDim.x * blockDim.x) + 2 * x + 1];
	uint32_t idx3 = d_orig_img[(2 * y + 1) * (gridDim.x * blockDim.x) + 2 * x];
	uint32_t idx4 = d_orig_img[(2 * y + 1) * (gridDim.x * blockDim.x) + 2 * x + 1];

	d_domains[half_id + rdblock_thread_id] = (uint8_t)((idx1 + idx2 + idx3 + idx4) >> 2);
}

/*
	各ドメインの総和，最小値最大値を計算する
	dim3 fc2dblock(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
	dim3 fc2dgrid(dblock_count / fc2dblock.z);
*/
__global__ void fcrr_domain_summimmax(uint8_t* d_domains,
								      uint32_t dblock_count,
									  uint32_t* dblock_sum,
								      uint32_t* dblock_min,
								      uint32_t* dblock_max) 
{
	//sum, min, maxの3種を保存する
	__shared__ uint32_t domain_summinmax[THREADBLOCK_MAX * 3];
	uint32_t dblock_id = blockIdx.x * blockDim.z + threadIdx.z;
	uint32_t pixel_total = blockDim.x * blockDim.y;
	uint32_t dblock_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t dblock_array_id = dblock_id * pixel_total + dblock_thread_id;

	uint32_t smem_block_id = threadIdx.z;
	uint32_t smem_thread_id = dblock_thread_id;
	uint32_t smem_array_sum_id = smem_block_id * (blockDim.x * blockDim.y) + smem_thread_id;
	uint32_t smem_array_min_id = smem_array_sum_id + THREADBLOCK_MAX;
	uint32_t smem_array_max_id = smem_array_min_id + THREADBLOCK_MAX;

	//if (smem_array_sum_id == THREADBLOCK_MAX) {
	//	printf("asdasfawdfja@opwjgf@paeo");

	//}

	uint8_t pixel = d_domains[dblock_array_id];
	//sum用
	domain_summinmax[smem_array_sum_id] = pixel;
	//min用
	domain_summinmax[smem_array_min_id] = pixel;
	//max用
	domain_summinmax[smem_array_max_id] = pixel;

	__syncthreads();

	for(int32_t i = (blockDim.x * blockDim.y) / 2; i > 0; i >>= 1){
		if(smem_thread_id < i){
			//sum
			domain_summinmax[smem_array_sum_id] += domain_summinmax[smem_array_sum_id + i];
			//min
			if(domain_summinmax[smem_array_min_id] > domain_summinmax[smem_array_min_id + i]){
				domain_summinmax[smem_array_min_id] = domain_summinmax[smem_array_min_id + i];
			}
			//max
			if (domain_summinmax[smem_array_max_id] < domain_summinmax[smem_array_max_id + i]) {
				domain_summinmax[smem_array_max_id] = domain_summinmax[smem_array_max_id + i];
			}
		}
		__syncthreads();
	}

	//保存
	if (dblock_thread_id == 0) {
		dblock_sum[dblock_id] = domain_summinmax[smem_array_sum_id];
		dblock_min[dblock_id] = domain_summinmax[smem_array_min_id];
		dblock_max[dblock_id] = domain_summinmax[smem_array_max_id];
	};
	__syncthreads();
}

/*
	各レンジの総和，最小値最大値を計算する
	dim3 fc2rblock(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
	dim3 fc2rgrid(rblock_count / fc2rblock.z);
*/
__global__ void fcrr_range_summimmax(uint8_t* d_ranges,
								   uint32_t rblock_count,
								   uint32_t* rblock_sum,
								   uint32_t* rblock_min,
								   uint32_t* rblock_max)
{
	//sum, min, maxの3種を保存する
	__shared__ uint32_t range_summinmax[THREADBLOCK_MAX * 3];
	uint32_t pixel_total = blockDim.x * blockDim.y;
	uint32_t rblock_id = blockIdx.x * blockDim.z + threadIdx.z;
	uint32_t rblock_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t rblock_array_id = rblock_id * pixel_total + rblock_thread_id;

	uint32_t smem_block_id = threadIdx.z;
	uint32_t smem_thread_id = rblock_thread_id;
	uint32_t smem_array_sum_id = smem_block_id * (blockDim.x * blockDim.y) + smem_thread_id;
	uint32_t smem_array_min_id = smem_array_sum_id + THREADBLOCK_MAX;
	uint32_t smem_array_max_id = smem_array_min_id + THREADBLOCK_MAX;

	uint8_t pixel = d_ranges[rblock_array_id];
	//sum用
	range_summinmax[smem_array_sum_id] = pixel;
	//min用
	range_summinmax[smem_array_min_id] = pixel;
	//max用
	range_summinmax[smem_array_max_id] = pixel;

	__syncthreads();

	for (int32_t i = (blockDim.x * blockDim.y) / 2; i > 0; i >>= 1) {
		if (smem_thread_id < i) {
			//sum
			range_summinmax[smem_array_sum_id] += range_summinmax[smem_array_sum_id + i];
			//min
			if (range_summinmax[smem_array_min_id] > range_summinmax[smem_array_min_id + i]) {
				range_summinmax[smem_array_min_id] = range_summinmax[smem_array_min_id + i];
			}
			//max
			if (range_summinmax[smem_array_max_id] < range_summinmax[smem_array_max_id + i]) {
				range_summinmax[smem_array_max_id] = range_summinmax[smem_array_max_id + i];
			}
		}
		__syncthreads();
	}

	//保存
	if (rblock_thread_id == 0) {
		//if(rblock_id == 2000){
		//	printf("d_range_value\n");
		//	for (int i = 0; blockDim.x > i; i++) {
		//		printf("%u, %u, %u, %u\n", (uint32_t)d_ranges[rblock_id * pixel_total + (i*4)], (uint32_t)d_ranges[rblock_id * pixel_total + (i * 4 + 1)], (uint32_t)d_ranges[rblock_id * pixel_total + (i * 4 + 2)], (uint32_t)d_ranges[rblock_id * pixel_total + (i * 4 + 3)]);
		//	}
		//	printf("---");
		//	printf("sum : %" PRIu32 "\n", range_summinmax[smem_array_sum_id]);
		//	printf("min : %" PRIu32 "\n", range_summinmax[smem_array_min_id]);
		//	printf("max : %" PRIu32 "\n", range_summinmax[smem_array_max_id]);
		//}
		rblock_sum[rblock_id] = range_summinmax[smem_array_sum_id];
		rblock_min[rblock_id] = range_summinmax[smem_array_min_id];
		rblock_max[rblock_id] = range_summinmax[smem_array_max_id];
	};
	__syncthreads();
}

/*
input:
	uint32_t* d_dblock_sum			:各ドメインブロックの総和
	uint32_t* d_dblock_min			:各ドメインブロックの最小値
	uint32_t* d_dblock_max			:各ドメインブロックの最大値
	uint32_t* d_rblock_sum			:各レンジブロックの総和
	uint32_t* d_rblock_min			:各レンジブロックの最小値
	uint32_t* d_rblock_max			:各レンジブロックの最大値
	uint32_t dr_block_pixel_total	:ブロック内の画素数
output:
	double* d_contrast_scaling		:各ドメインブロックの各レンジブロックに対する最適スケーリング
	uint32_t* d_brightness_shift	:各ドメインブロックの各レンジブロックに対する最適輝度シフト

call:
	dim3 fc3block(THREADBLOCK_MAX);
	dim3 fc3grid(dblock_count, rblock_count / THREADBLOCK_MAX);
	fc_calc_scale_n_shift<<<fc3grid, fc3block>>>
	//一つのスレッドブロックで複数のレンジブロックの最小値，最大値を計算する
*/
__global__ void fcrr_calc_scale_n_shift(uint32_t* d_dblock_sum,
									  uint32_t* d_dblock_min,
									  uint32_t* d_dblock_max,
									  uint32_t* d_rblock_sum,
									  uint32_t* d_rblock_min,
									  uint32_t* d_rblock_max,
									  uint32_t dr_block_pixel_total,
									  double* d_contrast_scaling,
									  uint32_t* d_brightness_shift)
{
	uint32_t dblock_id = blockIdx.x;
	uint32_t rblock_id = blockIdx.y * blockDim.x + threadIdx.x;
	uint32_t array_id =  blockIdx.x * (gridDim.y * blockDim.x) + rblock_id;

	//輝度シフト計算
	double shift = (((double)d_rblock_sum[rblock_id] - (double)d_dblock_sum[dblock_id])  / (double)dr_block_pixel_total);

	//if (array_id == 65000000) {
	//	printf("array max :  %" PRIu32 , (gridDim.x * gridDim.y * blockDim.x));
	//	printf("rblock_id  %" PRIu32 "dblock_id %" PRIu32 "\n", rblock_id, dblock_id);
	//	printf("rblock_id_sum[r]  %" PRIu32 "dblock_id_sum[d] %" PRIu32 "\n", d_rblock_sum[rblock_id], d_dblock_sum[dblock_id]);
	//	printf("rblock_id_sum[d]  %" PRIu32 "dblock_id_sum[d] %" PRIu32 "\n", d_rblock_sum[dblock_id], d_dblock_sum[dblock_id]);
	//	printf("contrast %lf, shift : %" PRIu32, d_contrast_scaling[array_id], d_brightness_shift[array_id]);
	//}


	if (shift < 0) {
		shift = 0;
	}
	else if (shift > 255) {
		shift = 255;
	}

	d_brightness_shift[array_id] = (uint32_t)shift;
	//コントラストスケーリング
	double d = (double)(d_dblock_max[dblock_id] - d_dblock_min[dblock_id]);
	double r = (double)(d_rblock_max[rblock_id] - d_rblock_min[rblock_id]);

	double raw_scaling = r / d;

	//uint8_t scaling_save = 0;
	for (double j = 0.0625; j <= 1; j += 0.0625) {
		if ((j - 0.0625) < raw_scaling && raw_scaling < j) {
			d_contrast_scaling[array_id] = j;
			break;
		}
		d_contrast_scaling[array_id] = 1;
	}



	//d_contrast_scaling[array_id] = raw_scaling;


	/*
		TODO 基本的に4bit内に縮小する必要が有る為，
		スケーリングの情報は圧縮して保持される必要がある
	*/

	//double min;
	//double max;
	//uint32_t scaling;
	//for (min = -0.03125, max = 0.03125, scaling = 0; scaling < 16; min += 0.0625, max += 0.0625, scaling++) {
	//	if (min < raw_scaling && raw_scaling < max){
	//		d_brightness_shift[array_id] = scaling;
	//		return;
	//	}
	//}
	////0.9625以上は全部15に・・・？
	//d_brightness_shift[array_id] = 0xF;
}

/*
	dim3 fc4block(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
	dim3 fc4grid(dblock_count , rblock_count / fc4block.z);
	fc_transform_n_calc_mse<<<fc4grid, fc4block>>>
*/
__global__ void fcrr_transform_n_calc_mse(uint8_t* d_domains,
										uint8_t* d_ranges,
										double* d_contrast_scaling,
										uint32_t* d_brightness_shift,
										uint32_t* d_mse,
										bool is_inner,
										uint32_t periphery,
										uint32_t rblock_cols,
										uint32_t rblock_rows)
{
	__shared__ uint32_t mse_all[THREADBLOCK_MAX];

	uint32_t drblock_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	uint32_t drblock_pixel_total = blockDim.x * blockDim.y;
	uint32_t dblock_id = blockIdx.x;
	uint32_t dblock_count = gridDim.x;
	uint32_t rblock_id = blockIdx.y * blockDim.z + threadIdx.z;
	uint32_t rblock_count = gridDim.y * blockDim.z;

	/*このスレッドが外周部分を担当しているかどうかのフラグ*/
	bool is_this_thread_outer = (rblock_id < rblock_cols * periphery/*上部*/ ||
								rblock_id >= rblock_count - rblock_cols * periphery/*下部*/ ||
								(rblock_id % rblock_cols) < periphery /*左部*/ ||
								(rblock_id % rblock_cols) >= rblock_cols - periphery)/*右部*/;

	if (is_inner == is_this_thread_outer) {
		return;
	}

	/*
	if (is_inner) {
		if (is_this_thread_outer){
			return;
		}
	}
	else {
		if (!is_this_thread_outer) {
			return;
		}
	}
	*/

	uint32_t array_id = dblock_id * rblock_count + rblock_id;

	uint32_t smem_array_id = threadIdx.z * drblock_pixel_total + drblock_thread_id;
	uint32_t smem_block_id = threadIdx.z * drblock_pixel_total;
	uint32_t smem_thread_id = drblock_thread_id;

	//このレンジにドメインに適応するscaling, shift
	uint32_t shift = d_brightness_shift[array_id];
	double scale = d_contrast_scaling[array_id];

	//レンジ
	uint8_t rpixel = d_ranges[rblock_id * drblock_pixel_total + drblock_thread_id];
	int32_t rpixel2 = (int32_t)rpixel * (int32_t)rpixel;
	double f_dpixel;
	for (int32_t rotate = 0; rotate < 8; rotate++) {
		if (rotate == 0){
			f_dpixel = (scale * (double)d_domains[dblock_id * (drblock_pixel_total) + drblock_thread_id]) + (double)shift;
		}
		else {
			if (blockDim.x == 4) {
				f_dpixel = (scale * (double)d_domains[dblock_id * (drblock_pixel_total) + (uint32_t)dc_affine_transform_size4[rotate - 1][drblock_thread_id]]) + (double)shift;
			}
			else if (blockDim.x == 8) {
				f_dpixel = (scale * (double)d_domains[dblock_id * (drblock_pixel_total) + (uint32_t)dc_affine_transform_size8[rotate - 1][drblock_thread_id]]) + (double)shift;
			}
			else if (blockDim.x == 16) {
				f_dpixel = (scale * (double)d_domains[dblock_id * (drblock_pixel_total) + (uint32_t)dc_affine_transform_size16[rotate - 1][drblock_thread_id]]) + (double)shift;
			}
		}
		if (f_dpixel > 255) {
			f_dpixel = 255;
		} else if (f_dpixel < 0) {
			f_dpixel = 0;
		}
		
		int32_t f_dpixel2 = (int32_t)f_dpixel * (int32_t)f_dpixel;
		int32_t diff = rpixel2 - f_dpixel2;
		diff = diff < 0 ? -diff : diff;
		uint32_t diff_abs = diff;
		mse_all[smem_array_id] = diff_abs;

		__syncthreads();

		for (int32_t j = drblock_pixel_total / 2; j > 0; j >>= 1) {
			if(smem_thread_id < j){
				mse_all[smem_array_id] += mse_all[smem_array_id + j];
			}
			__syncthreads();
		}
		if(smem_thread_id == 0){
			d_mse[rotate * dblock_count * rblock_count + dblock_id * rblock_count + rblock_id] = mse_all[smem_array_id];
		}
		__syncthreads();
	}
}

/*
	各レンジの二乗誤差が最小値であるドメイン(派生含む)のindexをリダクションで求め，各係数を保持する
	各スレッド毎に　最小となるmseを計算しておく
	dim3 fc5block(THREADBLOCK_MAX);
	dim3 fc5grid(rblock_count/ THREADBLOCK_MAX);
*/
__global__ void fcrr_save_min_mse(uint32_t dblock_cols,
								uint32_t dblock_rows,
								uint32_t blocksize,
								uint32_t* d_mse, 
								double* d_cotrast_scaling, 
								uint32_t* d_brightness_shift, 
								compress_data_part_reduce_ranges_gpu* d_compress_data_part_gpu,
								bool is_inner,
								uint32_t periphery,
								uint32_t rblock_cols,
								uint32_t rblock_rows) 
{
	uint32_t rblock_id = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t rblock_count = gridDim.x * blockDim.x;
	uint32_t dblock_count = dblock_cols * dblock_rows;
	
	uint32_t best_mse = UINT32_MAX;

	/*このスレッドが外周部分を担当しているかどうかのフラグ*/
	bool is_this_thread_outer = (rblock_id < rblock_cols * periphery/*上部*/ ||
								rblock_id >= rblock_count - rblock_cols * periphery/*下部*/ ||
								(rblock_id % rblock_cols) < periphery /*左部*/ ||
								(rblock_id % rblock_cols) >= rblock_cols - periphery)/*右部*/;

	if (is_inner == is_this_thread_outer) {
		return;
	}

	for (uint32_t dblock_y = 0; dblock_y < dblock_rows; dblock_y++) {
		for (uint32_t dblock_x = 0; dblock_x < dblock_cols; dblock_x++) {
			for (uint8_t rotate = 0; rotate < 7; rotate++) {
				uint32_t dblock_id = dblock_cols * dblock_y + dblock_x;
				uint32_t array_id = dblock_id * rblock_count + rblock_id;

				/*
					応急処置
				*/
				//bool medic = !((dblock_id * 4 == rblock_id) & (dblock_id * 4 == rblock_id + 1) & (dblock_id * 4 == rblock_id + 2) & (dblock_id * 4 == rblock_id + 3));
				bool medic = !(((rblock_id / blocksize) == dblock_id));

				if (d_mse[rotate * dblock_count * rblock_count + array_id] < best_mse & medic) {
					best_mse = d_mse[rotate * dblock_count * rblock_count + array_id];
					d_compress_data_part_gpu[rblock_id].rblock_id = rblock_id;
					d_compress_data_part_gpu[rblock_id].dblock_id = dblock_id;
					d_compress_data_part_gpu[rblock_id].rotate = rotate;
					d_compress_data_part_gpu[rblock_id].scale = d_cotrast_scaling[array_id];
					d_compress_data_part_gpu[rblock_id].shift = d_brightness_shift[array_id];
				}
				//if (rotate == 4 && dblock_x == 0 && dblock_y == 0 && blockIdx.x == 0 && threadIdx.x == 999) {
				//	printf("reached rotate == 3 value : %" PRIu32"\n best value : %" PRIu32"\n", d_mse[rotate * dblock_count * rblock_count + array_id], best_mse);
				//}
			}
		}
	}

}

/*
	CPU版
	全体が4x4，ブロックサイズが2x2の時，画素値の並びを1次以下の配列の並びにする
	{ 0, 1, 2, 3,
	  4, 5, 6, 6,
	  8, 9,10,11,
	 12,13,14,14} -> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
*/
void fcrr_img2array(cv::Mat img, uint8_t* img_array) {
	assert(img.isContinuous());
	img.convertTo(img, CV_8UC1);


	for (uint32_t y = 0; y < img.rows; y++) {
		for (uint32_t x = 0; x < img.cols; x++) {
			img_array[y*img.rows + x] = img.at<uint8_t>(y, x);
		}
	}
}

/*
input:
	cv::Mat img			: 一般的な画像の形式
	uint32_t blocksize	:ブロックの大きさ
	bool is_inner		:内部のブロックを処理するカーネルであるか
	periphery			:外周何周分を無視orのみ実行するか．
return;
	std::vector<ifs_transformer*> : 出力符号列

	is_innerがtrueの場合，外周periphery数値分だけ無視して演算を行う（スレッドは起動するが，内部は空）
	is_innerがfalseの場合，外周部分のみ実行する

	フラクタル圧縮の圧縮を行う一連のGPUカーネルを呼び
	imgを符号化する
*/
std::vector<ifs_transformer*> launch_reduce_ranges_compress_kernel(cv::Mat img, uint32_t blocksize, bool is_inner, uint32_t periphery)
{	
	/*
		0.前提条件
	*/

	//ドメイン・レンジブロックの１辺の長さ
	uint32_t dr_blocksize = blocksize;
	//ブロック1つが含む画素数
	uint32_t dr_block_pixel_total = dr_blocksize * dr_blocksize;
	//レンジブロックの辺当たりの数
	uint32_t rblock_cols = img.cols / dr_blocksize;
	uint32_t rblock_rows = img.rows / dr_blocksize;
	uint32_t rblock_count = rblock_cols * rblock_rows;
	//ドメインブロックの辺辺りの数
	uint32_t dblock_cols = rblock_cols >> 1;
	uint32_t dblock_rows = rblock_rows >> 1;
	uint32_t dblock_count = dblock_cols * dblock_rows;

	assert(blocksize == 4 || blocksize == 8 || blocksize == 16);
	assert((img.cols % blocksize) == 0);
	assert((img.rows % blocksize) == 0);
	assert(dblock_count % (THREADBLOCK_MAX / blocksize) == 0);
	assert(rblock_count % (THREADBLOCK_MAX / blocksize) == 0);
	assert(((rblock_count * dblock_count) % THREADBLOCK_MAX) == 0);
	assert(rblock_count < (65535 * THREADBLOCK_MAX));

	std::cout << "domain block count : " << dblock_count << std::endl;
	std::cout << "range block count : " << rblock_count << std::endl;

	/*
		1.ブロック変換・縮小変換
	*/

	//ブロック変換・縮小変換の為の起動スレッド数設定
	dim3 fc1block(dr_blocksize, dr_blocksize);
	dim3 fc1grid(rblock_cols, rblock_rows);
	std::cout << "reduction grid :" << " x = " << fc1grid.x << " y = " << fc1grid.y << std::endl;
	uint32_t orig_arraysize = img.total() * img.channels();
	uint8_t* h_orig_img = new uint8_t[orig_arraysize];
	uint8_t* d_orig_img;
	uint8_t* d_ranges;
	uint8_t* d_domains;
	
	CHECK(cudaMalloc((void**)&d_orig_img, sizeof(uint8_t) * orig_arraysize));
	CHECK(cudaMalloc((void**)&d_ranges, sizeof(uint8_t) * orig_arraysize));
	CHECK(cudaMalloc((void**)&d_domains, sizeof(uint8_t) * orig_arraysize >> 2));

	fcrr_img2array(img, h_orig_img);

	CHECK(cudaMemcpy(d_orig_img, h_orig_img, sizeof(uint8_t) * orig_arraysize, cudaMemcpyHostToDevice));
	fcrr_make_domains_n_ranges<<<fc1grid, fc1block>>>(d_orig_img, d_ranges, d_domains);
	CHECK(cudaDeviceSynchronize());

	//uint8_t* h_ranges = new uint8_t[orig_arraysize];
	//uint8_t* h_domains = new uint8_t[orig_arraysize >> 2];
	//CHECK(cudaMemcpy(h_ranges, d_ranges, sizeof(uint8_t) * orig_arraysize, cudaMemcpyDeviceToHost));
	//CHECK(cudaMemcpy(h_domains, d_domains, sizeof(uint8_t) * orig_arraysize >> 2, cudaMemcpyDeviceToHost));
	//show_img2(h_ranges, img.cols, img.rows, dr_blocksize);
	//show_img2(h_domains, img.cols >> 1, img.rows >> 1, dr_blocksize);

	/*
		2.ドメイン・レンジの総和・最小値最大値計算
	*/

	//１つのスレッドブロックで複数のドメインを処理する
	dim3 fc2dblock(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
	dim3 fc2dgrid(dblock_count / fc2dblock.z);

	uint32_t* d_dblock_sum;
	uint32_t* d_dblock_min;
	uint32_t* d_dblock_max;
	CHECK(cudaMalloc((void**)&d_dblock_sum, sizeof(uint32_t) * dblock_count));
	CHECK(cudaMalloc((void**)&d_dblock_min, sizeof(uint32_t) * dblock_count));
	CHECK(cudaMalloc((void**)&d_dblock_max, sizeof(uint32_t) * dblock_count));
	fcrr_domain_summimmax<<<fc2dgrid, fc2dblock>>>(d_domains, dblock_count, d_dblock_sum, d_dblock_min, d_dblock_max);

	//１つのスレッドブロックで複数のレンジを処理する
	dim3 fc2rblock(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
	dim3 fc2rgrid(rblock_count / fc2rblock.z);
	uint32_t* d_rblock_sum;
	uint32_t* d_rblock_min;
	uint32_t* d_rblock_max;
	CHECK(cudaMalloc((void**)&d_rblock_sum, sizeof(uint32_t) * rblock_count));
	CHECK(cudaMalloc((void**)&d_rblock_min, sizeof(uint32_t) * rblock_count));
	CHECK(cudaMalloc((void**)&d_rblock_max, sizeof(uint32_t) * rblock_count));
	fcrr_range_summimmax<<<fc2rgrid, fc2rblock>>>(d_ranges, rblock_count, d_rblock_sum, d_rblock_min, d_rblock_max);

	CHECK(cudaDeviceSynchronize());

	/*
		3.コントラストスケーリング・輝度シフト計算
	*/

	dim3 fc3block(THREADBLOCK_MAX);
	dim3 fc3grid(dblock_count, rblock_count / THREADBLOCK_MAX);

	double* d_contrast_scaling;
	uint32_t* d_brightness_shift;

	CHECK(cudaMalloc((void**)&d_contrast_scaling, sizeof(double) * dblock_count * rblock_count));
	CHECK(cudaMalloc((void**)&d_brightness_shift, sizeof(uint32_t) * dblock_count * rblock_count));
	//CHECK(cudaMalloc((void**)&d_adjust_domains_for_ranges, sizeof(uint32_t) * dblock_count * rblock_count * dr_block_pixel_total));

	//std::cout << "fc3grid : " << fc3grid.operator uint3 << "fc3block : " << fc3grid << std::endl;

	fcrr_calc_scale_n_shift<<<fc3grid, fc3block>>>(d_dblock_sum,
												 d_dblock_min,
												 d_dblock_max,
												 d_rblock_sum,
									     		 d_rblock_min,
												 d_rblock_max,
												 dr_block_pixel_total,
											     d_contrast_scaling,
											     d_brightness_shift);

	CHECK(cudaDeviceSynchronize());

	/*
		4.コントラストスケーリング・輝度シフト適用・回転・鏡像変換・二乗計算・MSE計算（リダクション）
	*/
	//ブロックサイズに応じたコンスタントメモリを初期化する
	init_fcrr_affine_transformer(dr_blocksize);
	CHECK(cudaDeviceSynchronize());
	dim3 fc4block(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
	dim3 fc4grid(dblock_count , rblock_count / fc4block.z);
	//各レンジの各ドメインの各回転変換後のMSEを保存しておく
	uint32_t* d_mse;
	CHECK(cudaMalloc((void**)&d_mse, sizeof(uint32_t) * rblock_count * dblock_count * 8));

	fcrr_transform_n_calc_mse<<<fc4grid, fc4block>>>(d_domains,
												     d_ranges,
												     d_contrast_scaling,
												     d_brightness_shift,
												     d_mse,
													 is_inner,
													 periphery,
													 rblock_cols,
													 rblock_rows);

	CHECK(cudaDeviceSynchronize());
	//std::cout << sizeof(uint32_t) * rblock_count * dblock_count * 8 << "byte" << std::endl;

	//uint32_t* h_mse = new uint32_t[rblock_count * dblock_count * 8];
	//CHECK(cudaMemcpy(h_mse, d_mse, sizeof(uint32_t) * rblock_count * dblock_count * 8, cudaMemcpyDeviceToHost));

	//std::cout << "nukiuti : " << h_mse[231] << std::endl;

	//delete[] h_mse;

	/*
		5.各レンジ毎最小MSE・index計算(リダクション諦め)
	*/

	dim3 fc5block(THREADBLOCK_MAX);
	dim3 fc5grid(rblock_count/ THREADBLOCK_MAX);

	//レンジの数だけ圧縮データを保存する
	compress_data_part_reduce_ranges_gpu* h_compress_data = new compress_data_part_reduce_ranges_gpu[rblock_count];
	compress_data_part_reduce_ranges_gpu* d_compress_data;
	CHECK(cudaMalloc((void**)&d_compress_data, sizeof(compress_data_part_reduce_ranges_gpu) * rblock_count));

	fcrr_save_min_mse<<<fc5grid, fc5block>>>(dblock_cols,
											 dblock_rows,
											 blocksize, d_mse, 
											 d_contrast_scaling,
											 d_brightness_shift, 
											 d_compress_data,
											 is_inner,
											 periphery,
											 rblock_cols,
											 rblock_rows);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(h_compress_data, d_compress_data, sizeof(compress_data_part_reduce_ranges_gpu) * rblock_count, cudaMemcpyDeviceToHost));

	/*
		6.HOST側のデータを共通の形式に加工
	*/

	CHECK(cudaDeviceSynchronize());

	std::vector<ifs_transformer*> ifs_data;

	for (int32_t i = 0; i < rblock_count; i++) {
		/*このスレッドが外周部分を担当しているかどうかのフラグ*/
		bool is_this_thread_outer = (i < rblock_cols * periphery/*上部*/ ||
			i >= rblock_count - rblock_cols * periphery/*下部*/ ||
			(i % rblock_cols) < periphery /*左部*/ ||
			(i % rblock_cols) >= rblock_cols - periphery)/*右部*/;

		if (is_inner == is_this_thread_outer) {
			continue;
		}

		//std::cout << h_compress_data[i].rotate << std::endl;
		ifs_transformer* c = new ifs_transformer();
		c->error = std::numeric_limits<double>::max();
		c->rblock_x = (h_compress_data[i].rblock_id % rblock_rows)*blocksize;
		c->rblock_y = (h_compress_data[i].rblock_id / rblock_rows)*blocksize;
		c->dblock_x = (h_compress_data[i].dblock_id % dblock_rows)*blocksize;
		c->dblock_y = (h_compress_data[i].dblock_id / dblock_rows)*blocksize;
		c->affine = h_compress_data[i].rotate;

		//std::cout << "h_comp:  " << h_compress_data[i].scale << std::endl;

		uint8_t scaling_save = 0;
		for (double j = 0.0625; j < 1; j += 0.0625) {
			//if ((j - 0.0625) <= h_compress_data[i].scale && h_compress_data[i].scale < j) {
			//	scaling_save = ((j) * 16);
			//	break;
			//}
			if (j == h_compress_data[i].scale) {
				break;
			}
			scaling_save++;
		}
		c->scaling = scaling_save;
		c->shift = h_compress_data[i].shift;
		c->blocksize = blocksize;
		ifs_data.push_back(c);
	}

	//std::cout << "just test : " << (int32_t)h_affine_transform_size4_1d[0] << std::endl;

	//delete[] h_ranges;
	//delete[] h_domains;

	/*
		L.後処理
	*/

	delete[] h_orig_img;
	delete[] h_compress_data;

	CHECK(cudaFree(d_orig_img));
	CHECK(cudaFree(d_ranges));
	CHECK(cudaFree(d_domains));

	CHECK(cudaFree(d_rblock_sum));
	CHECK(cudaFree(d_rblock_min));
	CHECK(cudaFree(d_rblock_max));
	
	CHECK(cudaFree(d_dblock_sum));
	CHECK(cudaFree(d_dblock_min));
	CHECK(cudaFree(d_dblock_max));

	CHECK(cudaFree(d_contrast_scaling));
	CHECK(cudaFree(d_brightness_shift));

	CHECK(cudaFree(d_mse));

	CHECK(cudaDeviceSynchronize());

	return ifs_data;
}

/*

	fc_make_range_n_domain<<<resize_grid, resize_block>>>(d_orig_img, d, d)


	dim3 resize_block(16, 16);
	dim3 resize_grid(((img.cols >> 1) + resize_block.x - 1) / resize_block.x, ((img.rows >> 1) + resize_block.y - 1) / resize_block.y);
	std::cout << "reduction grid :" << " x = " << resize_grid.x << " y = " << resize_grid.y << std::endl;
	
	//元画像を配列として表す場合の配列サイズ(単位byte)；
	uint32_t orig_img_array_size = img.total() * img.channels();
	uint8_t* h_orig_img = new uint8_t[orig_img_array_size];
	uint8_t* d_orig_img;
	CHECK(cudaMalloc((void**)&d_orig_img, sizeof(uint8_t) * orig_img_array_size));
	//縮小後画像を配列として表す場合の配列サイズ(単位byte);
	uint32_t stage1_size = img.total() * img.channels() >> 2;
	uint8_t* h_stage1 = new uint8_t[stage1_size];
	uint8_t* d_stage1;
	CHECK(cudaMalloc((void**)&d_stage1, sizeof(uint8_t) * stage1_size));

	//(img, h_orig_img);
	//show_img(h_orig_img, img.cols, img.rows);
	img2blockarray(img, h_orig_img, 32);
	show_img2(h_orig_img, img.rows, img.cols, 32);
	CHECK(cudaMemcpy(d_orig_img, h_orig_img, sizeof(uint8_t) * orig_img_array_size, cudaMemcpyHostToDevice));
	fc_resize2<<<resize_grid, resize_block>>>(d_orig_img, d_stage1, img.cols);
	//fc_resize<<<resize_grid, resize_block>>>(d_orig_img, d_stage1, img.cols);
	CHECK(cudaMemcpy(h_stage1, d_stage1, sizeof(uint8_t) * stage1_size, cudaMemcpyDeviceToHost));
	show_img2(h_stage1 ,(img.rows >> 1), (img.cols >> 1), 16);
	
	/*
		2.輝度シフト, コントラストスケーリング
	*/

	//dim3 resize_block(32, 32);
	//dim3 resize_grid(1, 1);
/*

	//test_kernel<<<1, 1>>>();
	CHECK(cudaDeviceSynchronize());

	delete[] h_orig_img;
	delete[] h_stage1;
*/