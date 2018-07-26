#pragma once

#include <opencv2/core.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
	ブロック内のスレッド数がこの値になるように努力する
	1024以下である必要がある
*/
#define THREADBLOCK_MAX 1024

struct compress_data_part_rgb_gpu {
	//対応するブロックのidを記憶しておく
	//実際に書き出す際は並べた順でidを識別できるので，レンジidはいらない
	uint32_t dblock_id;
	uint32_t rblock_id;
	double scale;
	uint8_t shift;
	//trueで鏡像
	bool mirror;
	//0 = 0', 1 = 90', 2 = 180', 3 = 270'
	uint8_t rotate;
	//どの層のdomainを用いるか
	//0 = R, 1 = G, 2 = B
	uint8_t color;
};

//void launch_rgb_compress_kernel(cv::Mat img, uint32_t blocksize);
