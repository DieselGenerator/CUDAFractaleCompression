//コンスタントメモリはグローバル変数にできない
//ということで別ファイルからアクセスできない為，deprecated
/*
	幾何変換の為の行列を生成する
	またGPUで扱う為にコンスタントメモリへの転送も行う
*/

#pragma once

#include <inttypes.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
	コンスタントメモリで幾何変換のindexを保持する
	affine_transform_sizen[0] = 90° 回転
	affine_transform_sizen[0] = 180°回転
	affine_transform_sizen[0] = 270°回転
	affine_transform_sizen[0] = 鏡像 0° 回転 
	affine_transform_sizen[0] = 鏡像 90°回転
	affine_transform_sizen[0] = 鏡像 180°回転
	affine_transform_sizen[0] = 鏡像 270°回転
*/
//extern __device__ uint8_t dc_affine_transform_size4_1d[7*16];
//
//extern __device__ uint8_t dc_affine_transform_size4[7][16];
//extern __device__ uint8_t dc_affine_transform_size8[7][64];
//extern __device__ uint8_t dc_affine_transform_size16[7][256];
//
//void init_affine_transformer(int size);
