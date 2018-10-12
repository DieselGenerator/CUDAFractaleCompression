#undef NDEBUG//assert用

#include <cassert>

#include "config.hpp"

/*
	デバッグに関する設定
*/
//重たいGPUの初回起動を緩和するための措置を実施するか
const bool config::enable_gpu_warmup = true;
//GPUの情報を出力するか
const bool config::enable_print_device_info = false;
//演算テスト（実験）を行うか
const bool config::enable_arithmetic_speedtest = false;
//事前に利用する為の幾何変換テーブルを生成しておくか
const bool config::enable_transform_generator = false;

/*
	CPU版OpenCVを用いた圧縮の設定
*/
//CPU版のOpenCVを利用した圧縮を行うか
const bool config::enable_cpu_opencv_compress = false;
//CPU版のOpenCVを利用した伸張を行うか（同方式の圧縮が行われる必要がある）
const bool config::enable_cpu_opencv_decompress = true;
//CPU版のOpenCVを利用した圧縮画像を展開したものを出力するか（同方式の伸張が行われる必要がある）
const bool config::enable_cpu_opencv_output = true;;
//圧縮を行う際の閾値
const double config::mse_threshold = 0;
//四分木分割を行うか
const bool config::enable_quartree_compress = true;
//range,domainの大きさの設定
const uint8_t config::range_size_min = 4;
const uint8_t config::range_size_max = 16;
const uint8_t config::domain_size_min = 8;
const uint8_t config::domain_size_max = 32;

/*
	GPU版OpenCVを用いた圧縮の設定(未完成)
*/
//GPU版のOpenCVを利用した圧縮を行うか
const bool config::enable_gpu_opencv_compress = false;

/*
	GPU版オリジナルカーネルを用いた圧縮の設定
*/
//GPUのオリジナルカーネルを利用した圧縮を行うか
const bool config::enable_gpu_compress = false;
//GPUのオリジナルカーネルを利用した伸張を行うか（同方式の圧縮が行われる必要がある）
const bool config::enable_gpu_decompress = true;
//GPUのオリジナルカーネルを利用した圧縮画像を展開したものを出力するか（同方式の伸張が行われる必要がある）
const bool config::enable_gpu_output = true;;

/*
	GPU版ドメイン減少カーネルを用いた圧縮の設定
*/
//GPUのドメイン減少カーネルを利用した圧縮を行うか
const bool config::enable_gpu_reduce_domains_compress = false;
//GPUのドメイン減少カーネルを利用した伸張を行うか（同方式の圧縮が行われる必要がある）
const bool config::enable_gpu_reduce_domains_decompress = true;
//GPUのドメイン減少カーネルを利用した圧縮画像を展開したものを出力するか（同方式の伸張が行われる必要がある）
const bool config::enable_gpu_reduce_domains_output = true;

/*
	GPU版レンジ減少カーネルを用いた圧縮の設定
*/
//GPUのレンジ減少カーネルを利用した圧縮を行うか
const bool config::enable_gpu_reduce_ranges_compress = true;
//GPUのレンジ減少カーネルを利用した伸張を行うか（同方式の圧縮が行われる必要がある）
const bool config::enable_gpu_reduce_ranges_decompress = true;
//GPUのレンジ減少カーネルを利用した圧縮画像を展開したものを出力するか（同方式の伸張が行われる必要がある）
const bool config::enable_gpu_reduce_ranges_output = true;

/*
	CPUでのlibJpegを用いた圧縮の設定
*/
const bool config::enable_jpeg_compress = false;
const bool config::enable_jpeg_output = false;
const bool config::enable_jpeg_repeat = false;
const bool config::enable_jpeg_output_statistics = false;

const bool config::show_decompress_process = false;
const bool config::output_decompress_process_image = false;
const bool config::output_decompress_image = true;

const bool config::show_demo = false;

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
