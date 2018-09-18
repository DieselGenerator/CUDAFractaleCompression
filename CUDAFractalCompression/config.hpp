/*
	各種設定を保持する
	値を変更しただけで再コンパイルさせたくないので実体と初期値をソースにおいた
*/

#pragma once

#include <cinttypes>

struct config {
	/*
		デバッグに関する設定
	*/
	static const bool enable_gpu_warmup;
	static const bool enable_print_device_info;
	static const bool enable_arithmetic_speedtest;
	static const bool enable_transform_generator;

	static const bool enable_cpu_opencv_compress;
	static const bool enable_cpu_opencv_decompress;

	static const bool enable_gpu_opencv_compress;

	static const bool enable_gpu_compress;
	static const bool enable_gpu_reduce_domains_compress;
	static const bool enable_gpu_reduce_ranges_compress;

	static const bool enable_jpeg_compress;
	static const bool enable_jpeg_output;
	static const bool enable_jpeg_repeat;
	static const bool enable_jpeg_output_statistics;

	static const bool show_decompress_process;
	static const bool output_decompress_process_image;
	static const bool output_decompress_image;

	static const bool show_demo;
	/*
		フラクタル圧縮に関する設定
	*/
	//constにしたのは間違いだった，可変値にする
	static const double config::mse_threshold;
	//四分岐分割を行わない場合はrange_size_maxの値で分割される
	static const bool enable_quartree_compress;

	//そのうち消す
	static const uint8_t range_size;
	static const uint8_t domain_size;

	static const uint8_t range_size_min;
	static const uint8_t range_size_max;

	static const uint8_t domain_size_min;
	static const uint8_t domain_size_max;

	/*
		CUDAに関する設定
	*/
	static const uint32_t thread_pre_block;
};

void config_assert();
