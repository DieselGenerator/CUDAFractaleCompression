#undef NDEBUG//assert用

#include <cassert>

#include "config.hpp"

/*
	デバッグに関する設定
*/
const bool config::enable_gpu_warmup = true;
const bool config::enable_print_device_info = false;
const bool config::enable_arithmetic_speedtest = false;
const bool config::enable_transform_generator = false;

const bool config::enable_cpu_opencv_compress = false;
const bool config::enable_cpu_opencv_decompress = false;

const bool config::enable_gpu_opencv_compress = false;

const bool config::enable_gpu_compress = false;
const bool config::enable_gpu_reduce_domains_compress = false;
const bool config::enable_gpu_reduce_ranges_compress = false;

const bool config::enable_jpeg_compress = true;
const bool config::enable_jpeg_output = true;
const bool config::enable_jpeg_repeat = true;
const bool config::enable_jpeg_output_statistics = true;

const bool config::show_decompress_process = true;
const bool config::output_decompress_process_image = false;
const bool config::output_decompress_image = true;

const bool config::show_demo = true;
/*
	フラクタル圧縮に関する設定
*/
const double config::mse_threshold = 0;
const bool config::enable_quartree_compress = true;

const uint8_t config::range_size_min = 4;
const uint8_t config::range_size_max = 16;

const uint8_t config::domain_size_min = 8;
const uint8_t config::domain_size_max = 32;

/*
	CUDAに関する設定
*/
const uint32_t thread_pre_block = 1024;

static_assert(config::range_size_min * 2 == config::domain_size_min, "range");

void config_assert() {
	assert(config::range_size_min * 2 == config::domain_size_min);
	assert(config::range_size_max * 2 == config::domain_size_max);
	assert(thread_pre_block <= 1024);
}
