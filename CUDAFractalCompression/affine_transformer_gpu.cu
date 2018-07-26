#include <iostream>

#include <inttypes.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "cuda_call_checker.cuh"
#include "affine_transformer_gpu.cuh"

/*
	�R���X�^���g�������ɓ]�����邽�߂̊􉽕ϊ���index��ێ�����
	affine_transform_sizen[0] = 90�� ��]
	affine_transform_sizen[0] = 180����]
	affine_transform_sizen[0] = 270����]
	affine_transform_sizen[0] = ���� 0�� ��] 
	affine_transform_sizen[0] = ���� 90����]
	affine_transform_sizen[0] = ���� 180����]
	affine_transform_sizen[0] = ���� 270����]
*/


//void init_affine_transformer(int size) {
//
//	uint8_t h_affine_transform_size4_1d[7 * 16] = {
//		//90
//		12,  8,  4,  0,
//		13,  9,  5,  1,
//		14, 10,  6,  2,
//		15, 11,  7,  3 ,
//		//180
//		15, 14, 13, 12,
//		11, 10,  9,  8,
//		7,  6,  5,  4,
//		3,  2,  1,  0 ,
//		//270
//		3,  7, 11, 15,
//		2,  6, 10, 14,
//		1,  5,  9, 13,
//		0,  4,  8, 12,
//		//mirror 0
//		3,  2,  1,  0,
//		7,  6,  5,  4,
//		11, 10,  9,  8,
//		15, 14, 13, 12,
//		//mirror 90
//		15, 11,  7,  3,
//		14, 10,  6,  2,
//		13,  9,  5,  1,
//		12,  8,  4,  0,
//		//mirror 180
//		12, 13, 14, 15,
//		8,  9, 10, 11,
//		4,  5,  6,  7,
//		0,  1,  2,  3 ,
//		//mirror 270
//		0,  4,  8, 12,
//		1,  5,  9, 13,
//		2,  6, 10, 14,
//		3,  7, 11, 15
//	};
//
//	uint8_t h_affine_transform_size4[7][16] = {
//		//90
//		{ 12,  8,  4,  0,
//		13,  9,  5,  1,
//		14, 10,  6,  2,
//		15, 11,  7,  3 },
//		//180
//		{ 15, 14, 13, 12,
//		11, 10,  9,  8,
//		7,  6,  5,  4,
//		3,  2,  1,  0 },
//		//270
//		{ 3,  7, 11, 15,
//		2,  6, 10, 14,
//		1,  5,  9, 13,
//		0,  4,  8, 12 },
//		//mirror 0
//		{ 3,  2,  1,  0,
//		7,  6,  5,  4,
//		11, 10,  9,  8,
//		15, 14, 13, 12 },
//		//mirror 90
//		{ 15, 11,  7,  3,
//		14, 10,  6,  2,
//		13,  9,  5,  1,
//		12,  8,  4,  0 },
//		//mirror 180
//		{ 12, 13, 14, 15,
//		8,  9, 10, 11,
//		4,  5,  6,  7,
//		0,  1,  2,  3 },
//		//mirror 270
//		{ 0,  4,  8, 12,
//		1,  5,  9, 13,
//		2,  6, 10, 14,
//		3,  7, 11, 15 }
//	};
//
//	uint8_t h_affine_transform_size8[7][64] = {
//		//90
//		{ 56, 48, 40, 32, 24, 16,  8,  0,
//		57, 49, 41, 33, 25, 17,  9,  1,
//		58, 50, 42, 34, 26, 18, 10,  2,
//		59, 51, 43, 35, 27, 19, 11,  3,
//		60, 52, 44, 36, 28, 20, 12,  4,
//		61, 53, 45, 37, 29, 21, 13,  5,
//		62, 54, 46, 38, 30, 22, 14,  6,
//		63, 55, 47, 39, 31, 23, 15,  7 },
//		//180
//		{ 63, 62, 61, 60, 59, 58, 57, 56,
//		55, 54, 53, 52, 51, 50, 49, 48,
//		47, 46, 45, 44, 43, 42, 41, 40,
//		39, 38, 37, 36, 35, 34, 33, 32,
//		31, 30, 29, 28, 27, 26, 25, 24,
//		23, 22, 21, 20, 19, 18, 17, 16,
//		15, 14, 13, 12, 11, 10,  9,  8,
//		7,  6,  5,  4,  3,  2,  1,  0 },
//		//270
//		{ 7, 15, 23, 31, 39, 47, 55, 63,
//		6, 14, 22, 30, 38, 46, 54, 62,
//		5, 13, 21, 29, 37, 45, 53, 61,
//		4, 12, 20, 28, 36, 44, 52, 60,
//		3, 11, 19, 27, 35, 43, 51, 59,
//		2, 10, 18, 26, 34, 42, 50, 58,
//		1,  9, 17, 25, 33, 41, 49, 57,
//		0,  8, 16, 24, 32, 40, 48, 56 },
//		//mirror
//		{ 7,  6,  5,  4,  3,  2,  1,  0,
//		15, 14, 13, 12, 11, 10,  9,  8,
//		23, 22, 21, 20, 19, 18, 17, 16,
//		31, 30, 29, 28, 27, 26, 25, 24,
//		39, 38, 37, 36, 35, 34, 33, 32,
//		47, 46, 45, 44, 43, 42, 41, 40,
//		55, 54, 53, 52, 51, 50, 49, 48,
//		63, 62, 61, 60, 59, 58, 57, 56 },
//		//mirror 90
//		{ 63, 55, 47, 39, 31, 23, 15,  7,
//		62, 54, 46, 38, 30, 22, 14,  6,
//		61, 53, 45, 37, 29, 21, 13,  5,
//		60, 52, 44, 36, 28, 20, 12,  4,
//		59, 51, 43, 35, 27, 19, 11,  3,
//		58, 50, 42, 34, 26, 18, 10,  2,
//		57, 49, 41, 33, 25, 17,  9,  1,
//		56, 48, 40, 32, 24, 16,  8,  0 },
//		//mirror 180
//		{ 56, 57, 58, 59, 60, 61, 62, 63,
//		48, 49, 50, 51, 52, 53, 54, 55,
//		40, 41, 42, 43, 44, 45, 46, 47,
//		32, 33, 34, 35, 36, 37, 38, 39,
//		24, 25, 26, 27, 28, 29, 30, 31,
//		16, 17, 18, 19, 20, 21, 22, 23,
//		8,  9, 10, 11, 12, 13, 14, 15,
//		0,  1,  2,  3,  4,  5,  6,  7 },
//		//mirror 270
//		{ 0,  8, 16, 24, 32, 40, 48, 56,
//		1,  9, 17, 25, 33, 41, 49, 57,
//		2, 10, 18, 26, 34, 42, 50, 58,
//		3, 11, 19, 27, 35, 43, 51, 59,
//		4, 12, 20, 28, 36, 44, 52, 60,
//		5, 13, 21, 29, 37, 45, 53, 61,
//		6, 14, 22, 30, 38, 46, 54, 62,
//		7, 15, 23, 31, 39, 47, 55, 63 }
//	};
//
//	uint8_t h_affine_transform_size16[7][256] = {
//		//90
//		{ 240,224,208,192,176,160,144,128,112, 96, 80, 64, 48, 32, 16,  0,
//		241,225,209,193,177,161,145,129,113, 97, 81, 65, 49, 33, 17,  1,
//		242,226,210,194,178,162,146,130,114, 98, 82, 66, 50, 34, 18,  2,
//		243,227,211,195,179,163,147,131,115, 99, 83, 67, 51, 35, 19,  3,
//		244,228,212,196,180,164,148,132,116,100, 84, 68, 52, 36, 20,  4,
//		245,229,213,197,181,165,149,133,117,101, 85, 69, 53, 37, 21,  5,
//		246,230,214,198,182,166,150,134,118,102, 86, 70, 54, 38, 22,  6,
//		247,231,215,199,183,167,151,135,119,103, 87, 71, 55, 39, 23,  7,
//		248,232,216,200,184,168,152,136,120,104, 88, 72, 56, 40, 24,  8,
//		249,233,217,201,185,169,153,137,121,105, 89, 73, 57, 41, 25,  9,
//		250,234,218,202,186,170,154,138,122,106, 90, 74, 58, 42, 26, 10,
//		251,235,219,203,187,171,155,139,123,107, 91, 75, 59, 43, 27, 11,
//		252,236,220,204,188,172,156,140,124,108, 92, 76, 60, 44, 28, 12,
//		253,237,221,205,189,173,157,141,125,109, 93, 77, 61, 45, 29, 13,
//		254,238,222,206,190,174,158,142,126,110, 94, 78, 62, 46, 30, 14,
//		255,239,223,207,191,175,159,143,127,111, 95, 79, 63, 47, 31, 15 },
//		//180
//		{ 255,254,253,252,251,250,249,248,247,246,245,244,243,242,241,240,
//		239,238,237,236,235,234,233,232,231,230,229,228,227,226,225,224,
//		223,222,221,220,219,218,217,216,215,214,213,212,211,210,209,208,
//		207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,
//		191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,
//		175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,160,
//		159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,144,
//		143,142,141,140,139,138,137,136,135,134,133,132,131,130,129,128,
//		127,126,125,124,123,122,121,120,119,118,117,116,115,114,113,112,
//		111,110,109,108,107,106,105,104,103,102,101,100, 99, 98, 97, 96,
//		95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80,
//		79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64,
//		63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
//		47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
//		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
//		15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0 },
//		//270
//		{ 15, 31, 47, 63, 79, 95,111,127,143,159,175,191,207,223,239,255,
//		14, 30, 46, 62, 78, 94,110,126,142,158,174,190,206,222,238,254,
//		13, 29, 45, 61, 77, 93,109,125,141,157,173,189,205,221,237,253,
//		12, 28, 44, 60, 76, 92,108,124,140,156,172,188,204,220,236,252,
//		11, 27, 43, 59, 75, 91,107,123,139,155,171,187,203,219,235,251,
//		10, 26, 42, 58, 74, 90,106,122,138,154,170,186,202,218,234,250,
//		9, 25, 41, 57, 73, 89,105,121,137,153,169,185,201,217,233,249,
//		8, 24, 40, 56, 72, 88,104,120,136,152,168,184,200,216,232,248,
//		7, 23, 39, 55, 71, 87,103,119,135,151,167,183,199,215,231,247,
//		6, 22, 38, 54, 70, 86,102,118,134,150,166,182,198,214,230,246,
//		5, 21, 37, 53, 69, 85,101,117,133,149,165,181,197,213,229,245,
//		4, 20, 36, 52, 68, 84,100,116,132,148,164,180,196,212,228,244,
//		3, 19, 35, 51, 67, 83, 99,115,131,147,163,179,195,211,227,243,
//		2, 18, 34, 50, 66, 82, 98,114,130,146,162,178,194,210,226,242,
//		1, 17, 33, 49, 65, 81, 97,113,129,145,161,177,193,209,225,241,
//		0, 16, 32, 48, 64, 80, 96,112,128,144,160,176,192,208,224,240 },
//		//mirror
//		{ 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
//		31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
//		47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
//		63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
//		79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64,
//		95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80,
//		111,110,109,108,107,106,105,104,103,102,101,100, 99, 98, 97, 96,
//		127,126,125,124,123,122,121,120,119,118,117,116,115,114,113,112,
//		143,142,141,140,139,138,137,136,135,134,133,132,131,130,129,128,
//		159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,144,
//		175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,160,
//		191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,
//		207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,
//		223,222,221,220,219,218,217,216,215,214,213,212,211,210,209,208,
//		239,238,237,236,235,234,233,232,231,230,229,228,227,226,225,224,
//		255,254,253,252,251,250,249,248,247,246,245,244,243,242,241,240 },
//		//mirror 90
//		{ 255,239,223,207,191,175,159,143,127,111, 95, 79, 63, 47, 31, 15,
//		254,238,222,206,190,174,158,142,126,110, 94, 78, 62, 46, 30, 14,
//		253,237,221,205,189,173,157,141,125,109, 93, 77, 61, 45, 29, 13,
//		252,236,220,204,188,172,156,140,124,108, 92, 76, 60, 44, 28, 12,
//		251,235,219,203,187,171,155,139,123,107, 91, 75, 59, 43, 27, 11,
//		250,234,218,202,186,170,154,138,122,106, 90, 74, 58, 42, 26, 10,
//		249,233,217,201,185,169,153,137,121,105, 89, 73, 57, 41, 25,  9,
//		248,232,216,200,184,168,152,136,120,104, 88, 72, 56, 40, 24,  8,
//		247,231,215,199,183,167,151,135,119,103, 87, 71, 55, 39, 23,  7,
//		246,230,214,198,182,166,150,134,118,102, 86, 70, 54, 38, 22,  6,
//		245,229,213,197,181,165,149,133,117,101, 85, 69, 53, 37, 21,  5,
//		244,228,212,196,180,164,148,132,116,100, 84, 68, 52, 36, 20,  4,
//		243,227,211,195,179,163,147,131,115, 99, 83, 67, 51, 35, 19,  3,
//		242,226,210,194,178,162,146,130,114, 98, 82, 66, 50, 34, 18,  2,
//		241,225,209,193,177,161,145,129,113, 97, 81, 65, 49, 33, 17,  1,
//		240,224,208,192,176,160,144,128,112, 96, 80, 64, 48, 32, 16,  0 },
//		//mirror 180
//		{ 240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,
//		224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,
//		208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,
//		192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,
//		176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
//		160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,
//		144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
//		128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,
//		112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,
//		96 ,97, 98, 99,100,101,102,103,104,105,106,107,108,109,110,111,
//		80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
//		64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
//		48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
//		32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
//		16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
//		0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
//		//mirror 270
//		{ 0, 16, 32, 48, 64, 80, 96,112,128,144,160,176,192,208,224,240,
//		1, 17, 33, 49, 65, 81, 97,113,129,145,161,177,193,209,225,241,
//		2, 18, 34, 50, 66, 82, 98,114,130,146,162,178,194,210,226,242,
//		3, 19, 35, 51, 67, 83, 99,115,131,147,163,179,195,211,227,243,
//		4, 20, 36, 52, 68, 84,100,116,132,148,164,180,196,212,228,244,
//		5, 21, 37, 53, 69, 85,101,117,133,149,165,181,197,213,229,245,
//		6, 22, 38, 54, 70, 86,102,118,134,150,166,182,198,214,230,246,
//		7, 23, 39, 55, 71, 87,103,119,135,151,167,183,199,215,231,247,
//		8, 24, 40, 56, 72, 88,104,120,136,152,168,184,200,216,232,248,
//		9, 25, 41, 57, 73, 89,105,121,137,153,169,185,201,217,233,249,
//		10, 26, 42, 58, 74, 90,106,122,138,154,170,186,202,218,234,250,
//		11, 27, 43, 59, 75, 91,107,123,139,155,171,187,203,219,235,251,
//		12, 28, 44, 60, 76, 92,108,124,140,156,172,188,204,220,236,252,
//		13, 29, 45, 61, 77, 93,109,125,141,157,173,189,205,221,237,253,
//		14, 30, 46, 62, 78, 94,110,126,142,158,174,190,206,222,238,254,
//		15, 31, 47, 63, 79, 95,111,127,143,159,175,191,207,223,239,255 }
//	};
//
//	if (size == 4) {
//		//CHECK(cudaMemcpyToSymbol(dc_affine_transform_size4_1d, h_affine_transform_size4_1d, sizeof(uint8_t) * 7 * 16));
//		//CHECK(cudaMemcpyToSymbol(dc_affine_transform_size4_1d, h_affine_transform_size4_1d, sizeof(uint8_t) * 7 * 16))
//		std::cout << "WTFFFFF" <<  (uint32_t)h_affine_transform_size4[3][0] << std::endl;
//		//CHECK(cudaMemcpyToSymbol(dc_affine_transform_size4_1d, h_affine_transform_size4_1d, sizeof(uint8_t) * 7 * 16));
//		//CHECK(cudaMemcpyToSymbol(dc_affine_transform_size4, h_affine_transform_size4, sizeof(uint8_t) * 7 * 16));
//		//CHECK(cudaMemcpy(dc_affine_transform_size4_1d, h_affine_transform_size4_1d, sizeof(uint8_t) * 7 * 16, cudaMemcpyHostToDevice));
//		std::cout << "size 4 copyed" << std::endl;
//		CHECK(cudaDeviceSynchronize());
//	}
//	//cudaMemC
//	else if (size == 8) {
//		CHECK(cudaMemcpyToSymbol(dc_affine_transform_size8, h_affine_transform_size8, sizeof(uint8_t) * 7 * 64));
//		std::cout << "size 8 copyed" << std::endl;
//	}
//	else if (size == 16) {
//		CHECK(cudaMemcpyToSymbol(dc_affine_transform_size16, h_affine_transform_size16, sizeof(uint8_t) * 7 * 256));
//		std::cout << "size 4 copyed" << std::endl;
//	}
//	CHECK(cudaDeviceSynchronize());
//}