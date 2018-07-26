//#undef NDEBUG
//
//#include <iostream>
//#include <cassert>
//
//#include <inttypes.h>
//
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>
//
//#include <opencv2/core.hpp>
//#include <opencv2/core/cuda.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudafeatures2d.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudawarping.hpp>
//#include <opencv2/cudafilters.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//
//#include "cuda_call_checker.cuh"
//#include "affine_transformer_gpu.cuh"
//#include "fractale_compressor_gpu_rgb.cuh"
//
///*
//	全体が4x4，ブロックサイズが2x2の時
//	{ 0, 1, 2, 3,     { 0, 1, 4, 5,
//	  4, 5, 6, 6,       2, 3, 6, 7,
//	  8, 9,10,11,       8, 9,12,13,
//	 12,13,14,14} ->   10,11,14,15}
//	 と並び替える，ブロックサイズはカーネル呼び出しで変更する
//	 fc_arrangement<<<(grid_x, grid_y), (block_x, block_y, n)>>>(i, o, size);
//	 各ブロック毎に
//*/
////いらん？
////__global__ void fc_arrangement(uint8_t* d_original_img, uint8_t* d_arrangement_img, uint32_t block_size){
////	extern uint8_t sm[];
////	
////	//ブロックの数
////	uint32_t blocks_num = gridDim.x;
////	//ブロックの大きさ
////	uint32_t block_total = blockDim.x * blockDim.y;
////
////
////
////	//ブロックの配列の先頭index
////	uint32_t dst_block_index = (blockIdx.y * blocks_num + blockIdx.x) * block_total;
////	//スレッドのindex
////	uint32_t dst_thread_index = threadIdx.y * blockDim.y + threadIdx.x;
////
////	//各ブロックの画素値を一時的に保持する
////	sm[dst_block_index + dst_thread_index];
////
////	__syncthreads();
////}
//
///*
//	一般的な画素の並びの配列に保存された画像を縦横1/2倍に縮小する
//	4点の平均値を算出するだけの方式
//*/
//__global__ void fc_resize(uint8_t* d_original_img,
//						  uint8_t* d_resize_img,
//						  uint32_t original_width) 
//{
//	uint32_t xx = threadIdx.x + blockIdx.x * blockDim.x;
//	uint32_t yy = threadIdx.y + blockIdx.y * blockDim.y;
//	uint32_t ix = xx << 1;
//	uint32_t iy = yy << 1;
//
//	uint32_t idx1 = d_original_img[(iy * original_width) + ix];
//	uint32_t idx2 = d_original_img[(iy * original_width) + ix + 1];
//	uint32_t idx3 = d_original_img[((iy + 1) * original_width) + ix];
//	uint32_t idx4 = d_original_img[((iy + 1) * original_width) + ix + 1];
//
//	d_resize_img[yy * (original_width >> 1)+ xx] = (uint8_t)((idx1 + idx2 + idx3 + idx4) >> 2);
//}
//
///*
//input:
//	uint8_t* d_orig_img		: 一般的な画像の形式
//output:
//	uint8_t* d_ranges	:ブロック化された画像の形式
//	uint8_t* d_domains	:ブロック化された画像の形式 画像サイズ1/2 ブロックサイズは同じ
//
//	フラクタル圧縮に必要な画像配列を生成する
//	カーネルを呼ぶ時のブロック数でドメイン（圧縮），レンジのブロックを決定する
//	fc_make_range_n_domain<<<grid, (block_x, block_y)>>>
//	のblock_x, block_yがブロックの大きさになる
//*/
//__global__ void fc_make_domains_n_ranges(uint8_t* d_orig_img, 
//									     uint8_t* d_ranges,
//									     uint8_t* d_domains){
//
//	uint32_t rdblock_id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y;
//	uint32_t rdblock_thread_id = blockDim.x * threadIdx.y + threadIdx.x;
//	uint32_t rdblock_array_id = rdblock_id + rdblock_thread_id;
//
//	uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
//	uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;
//	uint32_t normal_array_id = y * (gridDim.x * blockDim.x) + x;
//
//	d_ranges[rdblock_array_id] = d_orig_img[normal_array_id];
//
//	if( (blockIdx.x >= (gridDim.x >> 1) ) || ( blockIdx.y >= (gridDim.y >> 1) ) ){
//		return;
//	}
//
//	uint32_t half_id = (blockIdx.y * (gridDim.x >> 1) + blockIdx.x) * blockDim.x * blockDim.y;
//
//	uint32_t idx1 = d_orig_img[2 * y * (gridDim.x * blockDim.x) + 2 * x];
//	uint32_t idx2 = d_orig_img[2 * y * (gridDim.x * blockDim.x) + 2 * x + 1];
//	uint32_t idx3 = d_orig_img[(2 * y + 1) * (gridDim.x * blockDim.x) + 2 * x];
//	uint32_t idx4 = d_orig_img[(2 * y + 1) * (gridDim.x * blockDim.x) + 2 * x + 1];
//
//	d_domains[half_id + rdblock_thread_id] = (uint8_t)((idx1 + idx2 + idx3 + idx4) >> 2);
//}
//
///*
//	各ドメインの総和，最小値最大値を計算する
//*/
//__global__ void fc_domain_summimmax(uint8_t* d_domains,
//								    uint32_t dblock_count,
//									uint32_t* dblock_sum,
//								    uint32_t* dblock_min,
//								    uint32_t* dblock_max) 
//{
//	//sum, min, maxの3種を保存する
//	__shared__ uint32_t domain_summinmax[THREADBLOCK_MAX * 3];
//	uint32_t dblock_id = blockIdx.x * blockDim.z + threadIdx.z;
//	uint32_t dblock_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
//	uint32_t dblock_array_id = dblock_id + dblock_thread_id;
//
//	uint32_t smem_block_id = threadIdx.z;
//	uint32_t smem_thread_id = dblock_thread_id;
//	uint32_t smem_array_sum_id = smem_block_id * (blockDim.x * blockDim.y) + smem_thread_id;
//	uint32_t smem_array_min_id = smem_array_sum_id + THREADBLOCK_MAX;
//	uint32_t smem_array_max_id = smem_array_min_id + THREADBLOCK_MAX;
//
//	if (smem_array_sum_id == THREADBLOCK_MAX) {
//		printf("asdasfawdfja@opwjgf@paeo");
//
//	}
//
//	uint8_t pixel = d_domains[dblock_array_id];
//	//sum用
//	domain_summinmax[smem_array_sum_id] = pixel;
//	//min用
//	domain_summinmax[smem_array_min_id] = pixel;
//	//max用
//	domain_summinmax[smem_array_max_id] = pixel;
//
//	__syncthreads();
//
//	for(int32_t i = (blockDim.x * blockDim.y) / 2; i > 0; i >>= 1){
//		if(smem_thread_id < i){
//			//sum
//			domain_summinmax[smem_array_sum_id] += domain_summinmax[smem_array_sum_id + i];
//			//min
//			if(domain_summinmax[smem_array_min_id] > domain_summinmax[smem_array_min_id + i]){
//				domain_summinmax[smem_array_min_id] = domain_summinmax[smem_array_min_id + i];
//			}
//			//max
//			if (domain_summinmax[smem_array_max_id] < domain_summinmax[smem_array_max_id + i]) {
//				domain_summinmax[smem_array_max_id] = domain_summinmax[smem_array_max_id + i];
//			}
//		}
//		__syncthreads();
//	}
//
//	//保存
//	if (dblock_thread_id == 0) {
//		dblock_sum[dblock_id] = domain_summinmax[smem_array_sum_id];
//		dblock_min[dblock_id] = domain_summinmax[smem_array_min_id];
//		dblock_max[dblock_id] = domain_summinmax[smem_array_max_id];
//	};
//}
//
///*
//	各レンジの総和，最小値最大値を計算する
//
//
//*/
//__global__ void fc_range_summimmax(uint8_t* d_ranges,
//								   uint32_t rblock_count,
//								   uint32_t* rblock_sum,
//								   uint32_t* rblock_min,
//								   uint32_t* rblock_max)
//{
//	//sum, min, maxの3種を保存する
//	__shared__ uint32_t range_summinmax[THREADBLOCK_MAX * 3];
//	uint32_t rblock_id = blockIdx.x * blockDim.z + threadIdx.z;
//	uint32_t rblock_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
//	uint32_t rblock_array_id = rblock_id + rblock_thread_id;
//
//	uint32_t smem_block_id = threadIdx.z;
//	uint32_t smem_thread_id = rblock_thread_id;
//	uint32_t smem_array_sum_id = smem_block_id * (blockDim.x * blockDim.y) + smem_thread_id;
//	uint32_t smem_array_min_id = smem_array_sum_id + THREADBLOCK_MAX;
//	uint32_t smem_array_max_id = smem_array_min_id + THREADBLOCK_MAX;
//
//	uint8_t pixel = d_ranges[rblock_array_id];
//	//sum用
//	range_summinmax[smem_array_sum_id] = pixel;
//	//min用
//	range_summinmax[smem_array_min_id] = pixel;
//	//max用
//	range_summinmax[smem_array_max_id] = pixel;
//
//	__syncthreads();
//
//	for (int32_t i = (blockDim.x * blockDim.y) / 2; i > 0; i >>= 1) {
//		if (smem_thread_id < i) {
//			//sum
//			range_summinmax[smem_array_sum_id] += range_summinmax[smem_array_sum_id + i];
//			//min
//			if (range_summinmax[smem_array_min_id] > range_summinmax[smem_array_min_id + i]) {
//				range_summinmax[smem_array_min_id] = range_summinmax[smem_array_min_id + i];
//			}
//			//max
//			if (range_summinmax[smem_array_max_id] < range_summinmax[smem_array_max_id + i]) {
//				range_summinmax[smem_array_max_id] = range_summinmax[smem_array_max_id + i];
//			}
//		}
//		__syncthreads();
//	}
//
//	//保存
//	if (rblock_thread_id == 0) {
//		//if(rblock_id == 0){
//		//	printf("sum : %" PRIu32 "\n", range_summinmax[smem_array_sum_id]);
//		//	printf("min : %" PRIu32 "\n", range_summinmax[smem_array_min_id]);
//		//	printf("max : %" PRIu32 "\n", range_summinmax[smem_array_max_id]);
//		//}
//		rblock_sum[rblock_id] = range_summinmax[smem_array_sum_id];
//		rblock_min[rblock_id] = range_summinmax[smem_array_min_id];
//		rblock_max[rblock_id] = range_summinmax[smem_array_max_id];
//	};
//}
//
///*
//input:
//	uint32_t* d_dblock_sum			:各ドメインブロックの総和
//	uint32_t* d_dblock_min			:各ドメインブロックの最小値
//	uint32_t* d_dblock_max			:各ドメインブロックの最大値
//	uint32_t* d_rblock_sum			:各レンジブロックの総和
//	uint32_t* d_rblock_min			:各レンジブロックの最小値
//	uint32_t* d_rblock_max			:各レンジブロックの最大値
//	uint32_t dr_block_pixel_total	:ブロック内の画素数
//output:
//	double* d_contrast_scaling		:各ドメインブロックの各レンジブロックに対する最適スケーリング
//	uint32_t* d_brightness_shift	:各ドメインブロックの各レンジブロックに対する最適輝度シフト
//
//call:
//	dim3 fc3block(THREADBLOCK_MAX);
//	dim3 fc3grid(dblock_count, rblock_count / THREADBLOCK_MAX);
//	fc_calc_scale_n_shift<<<fc3grid, fc3block>>>
//	//一つのスレッドブロックで複数のレンジブロックの最小値，最大値を計算する
//*/
//__global__ void fc_calc_scale_n_shift(uint32_t* d_dblock_sum,
//									  uint32_t* d_dblock_min,
//									  uint32_t* d_dblock_max,
//									  uint32_t* d_rblock_sum,
//									  uint32_t* d_rblock_min,
//									  uint32_t* d_rblock_max,
//									  uint32_t dr_block_pixel_total,
//									  double* d_contrast_scaling,
//									  uint32_t* d_brightness_shift)
//{
//	uint32_t dblock_id = blockIdx.x;
//	uint32_t rblock_id = blockIdx.y * blockDim.x + threadIdx.x;
//	uint32_t array_id =  blockIdx.x * (gridDim.y * blockDim.x) + rblock_id;
//
//	//輝度シフト計算
//	int32_t shift = ((int32_t)d_dblock_sum[dblock_id] - (int32_t)d_rblock_sum[rblock_id])  / dr_block_pixel_total;
//	d_brightness_shift[array_id] = shift;
//	//コントラストスケーリング
//	double d = (double)(d_dblock_max[dblock_id] - d_dblock_min[dblock_id]);
//	double r = (double)(d_rblock_max[rblock_id] - d_rblock_min[rblock_id]);
//
//	double raw_scaling = r / d;
//	d_contrast_scaling[array_id] = raw_scaling;
//
//	/*
//		TODO 基本的に4bit内に縮小する必要が有る為，
//		スケーリングの情報は圧縮して保持される必要がある
//	*/
//
//	//double min;
//	//double max;
//	//uint32_t scaling;
//	//for (min = -0.03125, max = 0.03125, scaling = 0; scaling < 16; min += 0.0625, max += 0.0625, scaling++) {
//	//	if (min < raw_scaling && raw_scaling < max){
//	//		d_brightness_shift[array_id] = scaling;
//	//		return;
//	//	}
//	//}
//	////0.9625以上は全部15に・・・？
//	//d_brightness_shift[array_id] = 0xF;
//}
//
///*
//	dim3 fc4block(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
//	dim3 fc4grid(dblock_count , rblock_count / fc4block.z);
//	fc_transform_n_calc_mse<<<fc4grid, fc4block>>>
//*/
//__global__ void fc_transform_n_calc_mse(uint8_t* d_domains,
//										uint8_t* d_ranges,
//										double* d_contrast_scaling,
//										uint32_t* d_brightness_shift,
//										uint32_t* mse)
//{
//	__shared__ uint32_t mse_all[THREADBLOCK_MAX];
//
//	uint32_t drblock_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
//	uint32_t drblock_pixel_total = blockDim.x * blockDim.y;
//	uint32_t dblock_id = blockIdx.x;
//	uint32_t dblock_count = gridDim.x;
//	uint32_t rblock_id = blockIdx.y * blockDim.z + threadIdx.z;
//	uint32_t rblock_count = gridDim.y * blockDim.z;
//
//	uint32_t array_id = dblock_id * rblock_count + rblock_id;
//
//	uint32_t smem_array_id = threadIdx.z * drblock_pixel_total + drblock_thread_id;
//	uint32_t smem_block_id = threadIdx.z * drblock_pixel_total;
//	uint32_t smem_thread_id = drblock_thread_id;
//
//	//このレンジにドメインに適応するscaling, shift
//	uint32_t shift = d_brightness_shift[array_id];
//	double scale = d_contrast_scaling[array_id];
//
//	double f_d_p = scale * (double)d_domains[dblock_id + drblock_thread_id] + (double)shift;
//
//	if(f_d_p < 0){
//		f_d_p = -f_d_p;
//	}
//	//uint8_tにキャストした時点で最大値は255に固定される
//	uint8_t fixed_dpixel = (uint8_t)f_d_p;
//	int32_t fixed_dpixel2 = (int32_t)fixed_dpixel * (int32_t)fixed_dpixel;
//	//オリジナルレンジ
//	uint8_t rpixel = d_ranges[rblock_id + drblock_thread_id];
//
//	for(int32_t rotate = 0; rotate < 8; rotate++){
//		int32_t fixed_dpixel2 = (int32_t)fixed_dpixel * (int32_t)fixed_dpixel;
//		int32_t rpixel2 = (int32_t)rpixel * (int32_t)rpixel;
//		int32_t diff = fixed_dpixel2 - rpixel2;
//		diff = diff < 0 ? -diff : diff;
//		uint32_t diff_abs = diff;
//		mse_all[smem_array_id] = diff_abs;
//
//		for (int32_t j = drblock_pixel_total / 2; j > 0; j >>= 1) {
//			if(smem_thread_id < j){
//				mse_all[smem_array_id] += mse_all[smem_array_id + j];
//			}
//			__syncthreads();
//		}
//		if(smem_thread_id == 0){
//			mse[rotate * dblock_count * rblock_count + dblock_id * rblock_count + rblock_id] = mse_all[smem_array_id];
//		}
//		if (rotate < 7) {
//			if (blockDim.x == 4) {
//				rpixel = d_ranges[rblock_id + dc_affine_transform_size4[rotate][drblock_thread_id]];
//			}
//			else if (blockDim.x == 8) {
//				rpixel = d_ranges[rblock_id + dc_affine_transform_size8[rotate][drblock_thread_id]];
//			}
//			else if (blockDim.x == 16) {
//				rpixel = d_ranges[rblock_id + dc_affine_transform_size16[rotate][drblock_thread_id]];
//			}
//		}
//		__syncthreads();
//	}
//}
//
///*
//	各レンジの二乗誤差が最小値であるドメイン(派生含む)のindexをリダクションで求め，各係数を保持する
//*/
//__global__ void fc_save_min_mse(uint32_t* d_mse, 
//								double* d_cotrast_scaling, 
//								uint32_t* d_brightness_shift, 
//								compress_data_part_rgb_gpu* d_compress_data_part_gpu) 
//{
//	
//}
//
///*
//	//
//	画像を縦横1/2倍に縮小する
//	4点の平均値を算出するだけの方式
//	カーネルでブロックサイズを合わせる
//	元ブロックサイズを
//	元カネールの
//	fc_resize2<<<(grid_x, grid_y), (block_x, block_y, n)>>>(i, o, size);
//*/
//__global__ void fc_resize2(uint8_t* d_original_img, uint8_t* d_resize_img, uint32_t original_width) {
//	//blockDim.x, blockDim.yはリサイズ後のブロックサイズ
//	//blockの数自体は変換前後で一定
//	uint32_t blocks_num = gridDim.x;//original_width / blockDim.x;
//
//	//resize後ブロックの大きさ
//	uint32_t resize_block_total = blockDim.x * blockDim.y;
//	//resize後ブロックの配列の先頭index
//	uint32_t resize_block_index = (blockIdx.y * blocks_num + blockIdx.x) * resize_block_total;
//	//resize後ブロック内スレッドのindex
//	uint32_t resize_thread_index = threadIdx.y * blockDim.y + threadIdx.x;
//
//	//元ブロックの大きさ
//	uint32_t orig_block_total = resize_block_total << 2;
//	//元ブロックの大きさの配列の先頭index
//	uint32_t orig_block_index = (blockIdx.y * blocks_num + blockIdx.x) * orig_block_total;
//	//resize後ブロック内スレッドのindex1
//	uint32_t orig_thread_index1 = (threadIdx.y << 1) * (blockDim.y << 1) + (threadIdx.x << 1);
//	//resize後ブロック内スレッドのindex2
//	uint32_t orig_thread_index2 = (threadIdx.y << 1) * (blockDim.y << 1) + (threadIdx.x << 1) + 1;
//	//resize後ブロック内スレッドのindex3
//	uint32_t orig_thread_index3 = ((threadIdx.y << 1) + 1) * (blockDim.y << 1) + (threadIdx.x << 1);
//	//resize後ブロック内スレッドのindex4
//	uint32_t orig_thread_index4 = ((threadIdx.y << 1) + 1) * (blockDim.y << 1) + (threadIdx.x << 1) + 1;
//
//	//resize後ブロック内スレッド
//	uint32_t idx1 = d_original_img[orig_block_index + orig_thread_index1];
//	uint32_t idx2 = d_original_img[orig_block_index + orig_thread_index2];
//	uint32_t idx3 = d_original_img[orig_block_index + orig_thread_index3];
//	uint32_t idx4 = d_original_img[orig_block_index + orig_thread_index4];
//
//	d_resize_img[resize_block_index + resize_thread_index] = (uint8_t)((idx1 + idx2 + idx3 + idx4) >> 2);
//}
//
///*
//	全体が4x4，ブロックサイズが2x2の時
//	{ 0, 1, 2, 3,
//	  4, 5, 6, 6,
//	  8, 9,10,11,
//	 12,13,14,14}
//	のような画素値の並びのグレースケールの配列画像をテスト表示する
//	全画素を代入していくので速度は遅い，テスト用
//*/
//void show_img(uint8_t* img_array, uint32_t width, uint32_t height){
//	cv::Mat mat(width, height, CV_8U);
//	for (uint32_t y = 0; y < height; y++) {
//		for (uint32_t x = 0; x < width; x++) {
//			mat.at<uint8_t>(y, x) = img_array[y*width + x];
//		}
//	}
//	cv::namedWindow("show_img", cv::WINDOW_AUTOSIZE);
//	cv::imshow("show_img", mat);
//	cv::waitKey(0);
//	cv::destroyAllWindows();
//}
//
///*
//	全体が4x4，ブロックサイズが2x2の時
//	{ 0, 1, 4, 5,
//	  2, 3, 6, 7,
//	  8, 9,12,13,
//	 10,11,14,15}
//	のような画素値の並びのグレースケールの配列画像をテスト表示する
//	全画素を代入していくので速度は遅い，テスト用
//	block_size == block_height == block_width
//*/
//void show_img2(uint8_t* img_array, uint32_t width, uint32_t height, uint32_t block_size) {
//	cv::Mat mat(width, height, CV_8U);
//
//	//横方向のレンジの数
//	uint32_t range_x_n = width / block_size;
//	//縦方向のレンジの数
//	uint32_t range_y_n = height / block_size;
//
//	uint32_t index = 0;
//	for(uint32_t y = 0; y < range_y_n * block_size; y += block_size){
//		for (uint32_t x = 0; x < range_x_n * block_size; x += block_size) {
//			//各ブロック
//			for (uint32_t block_rows = 0; block_rows < block_size; block_rows++) {
//				for (uint32_t block_cols = 0; block_cols < block_size; block_cols++) {
//					mat.at<uint8_t>(y + block_rows, x + block_cols) = img_array[index];
//					index++;
//				}
//			}
//		}
//	}
//
//	cv::namedWindow("show_img", cv::WINDOW_AUTOSIZE);
//	cv::imshow("show_img", mat);
//	cv::waitKey(0);
//	cv::destroyAllWindows();
//}
//
//
///*
//	CPU版
//	全体が4x4，ブロックサイズが2x2の時，画素値の並びを1次以下の配列の並びにする
//	{ 0, 1, 2, 3,
//	  4, 5, 6, 6,
//	  8, 9,10,11,
//	 12,13,14,14} -> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
//*/
//void img2array(cv::Mat img, uint8_t* img_array) {
//	assert(img.isContinuous());
//	img.convertTo(img, CV_8UC1);
//
//
//	for (uint32_t y = 0; y < img.rows; y++) {
//		for (uint32_t x = 0; x < img.cols; x++) {
//			img_array[y*img.rows + x] = img.at<uint8_t>(y, x);
//		}
//	}
//}
//
///*
//	CPU版
//	全体が4x4，ブロックサイズが2x2の時，画素値の並びを1次以下の配列の並びにする
//	{ 0, 1, 2, 3,
//	  4, 5, 6, 7,
//	  8, 9,10,11,
//	 12,13,14,15} -> {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}
//*/
//void img2blockarray(cv::Mat img, uint8_t* img_array, uint32_t block_size){
//	assert(img.isContinuous());
//	assert((img.cols % block_size) == 0);
//	assert((img.rows % block_size) == 0);
//	img.convertTo(img, CV_8UC1);
//
//	//横方向のレンジの数
//	uint32_t range_x_n = img.cols / block_size;
//	//縦方向のレンジの数
//	uint32_t range_y_n = img.rows / block_size;
//	std::cout << "tatal : "<<img.total() << std::endl;
//	uint32_t index = 0;
//	for (uint32_t y = 0; y < range_y_n * block_size; y += block_size) {
//		for (uint32_t x = 0; x < range_x_n * block_size; x += block_size) {
//			//各ブロック
//			
//			for (uint32_t block_rows = 0; block_rows < block_size; block_rows++) {
//				for (uint32_t block_cols = 0; block_cols < block_size; block_cols++) {
//					
//					//if(index % 1000 == 0)std::cout << index << std::endl;
//					img_array[index] = img.at<uint8_t>(y + block_rows, x + block_cols);
//					index++;
//				}
//			}
//		}
//	}
//
//}
//
//void launch_rgb_compress_kernel(cv::Mat img, uint32_t blocksize)
//{	
//	/*
//		0.前提条件
//	*/
//
//	//ドメイン・レンジブロックの１辺の長さ
//	uint32_t dr_blocksize = blocksize;
//	//ブロック1つが含む画素数
//	uint32_t dr_block_pixel_total = dr_blocksize * dr_blocksize;
//	//レンジブロックの辺当たりの数
//	uint32_t rblock_cols = img.cols / dr_blocksize;
//	uint32_t rblock_rows = img.rows / dr_blocksize;
//	uint32_t rblock_count = rblock_cols * rblock_rows;
//	//ドメインブロックの辺辺りの数
//	uint32_t dblock_cols = rblock_cols >> 1;
//	uint32_t dblock_rows = rblock_rows >> 1;
//	uint32_t dblock_count = dblock_cols * dblock_rows;
//
//	assert(blocksize == 4 || blocksize == 8 || blocksize == 16);
//	assert((img.cols % blocksize) == 0);
//	assert((img.rows % blocksize) == 0);
//	assert(dblock_count % (THREADBLOCK_MAX / blocksize) == 0);
//	assert(rblock_count % (THREADBLOCK_MAX / blocksize) == 0);
//	assert(((rblock_count * dblock_count) % THREADBLOCK_MAX) == 0);
//	assert(rblock_count < (65535 * THREADBLOCK_MAX));
//
//	std::cout << "domain block count : " << dblock_count << std::endl;
//	std::cout << "range block count : " << rblock_count << std::endl;
//
//	/*
//		1.ブロック変換・縮小変換
//	*/
//	//ブロック変換・縮小変換の為の起動スレッド数設定
//	dim3 fc1block(dr_blocksize, dr_blocksize);
//	dim3 fc1grid(rblock_cols, rblock_rows);
//	std::cout << "reduction grid :" << " x = " << fc1grid.x << " y = " << fc1grid.y << std::endl;
//	uint32_t orig_arraysize = img.total() * img.channels();
//	uint8_t* h_orig_img = new uint8_t[orig_arraysize];
//	uint8_t* d_orig_img;
//	uint8_t* d_ranges;
//	uint8_t* d_domains;
//	
//	CHECK(cudaMalloc((void**)&d_orig_img, sizeof(uint8_t) * orig_arraysize));
//	CHECK(cudaMalloc((void**)&d_ranges, sizeof(uint8_t) * orig_arraysize));
//	CHECK(cudaMalloc((void**)&d_domains, sizeof(uint8_t) * orig_arraysize >> 2));
//
//	img2array(img, h_orig_img);
//
//	CHECK(cudaMemcpy(d_orig_img, h_orig_img, sizeof(uint8_t) * orig_arraysize, cudaMemcpyHostToDevice));
//	fc_make_domains_n_ranges<<<fc1grid, fc1block>>>(d_orig_img, d_ranges, d_domains);
//	CHECK(cudaDeviceSynchronize());
//	//uint8_t* h_ranges = new uint8_t[orig_arraysize];
//	//uint8_t* h_domains = new uint8_t[orig_arraysize >> 2];
//	//CHECK(cudaMemcpy(h_ranges, d_ranges, sizeof(uint8_t) * orig_arraysize, cudaMemcpyDeviceToHost));
//	//CHECK(cudaMemcpy(h_domains, d_domains, sizeof(uint8_t) * orig_arraysize >> 2, cudaMemcpyDeviceToHost));
//	//show_img2(h_ranges, img.cols, img.rows, dr_blocksize);
//	//show_img2(h_domains, img.cols >> 1, img.rows >> 1, dr_blocksize);
//
//	/*
//		2.ドメイン・レンジの総和・最小値最大値計算
//	*/
//	//１つのスレッドブロックで複数のドメインを処理する
//	dim3 fc2dblock(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
//	dim3 fc2dgrid(dblock_count / fc2dblock.z);
//
//	uint32_t* d_dblock_sum;
//	uint32_t* d_dblock_min;
//	uint32_t* d_dblock_max;
//	CHECK(cudaMalloc((void**)&d_dblock_sum, sizeof(uint32_t) * dblock_count));
//	CHECK(cudaMalloc((void**)&d_dblock_min, sizeof(uint32_t) * dblock_count));
//	CHECK(cudaMalloc((void**)&d_dblock_max, sizeof(uint32_t) * dblock_count));
//	fc_domain_summimmax<<<fc2dgrid, fc2dblock>>>(d_domains, dblock_count, d_dblock_sum, d_dblock_min, d_dblock_max);
//
//	//１つのスレッドブロックで複数のレンジを処理する
//	dim3 fc2rblock(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
//	dim3 fc2rgrid(rblock_count / fc2rblock.z);
//	uint32_t* d_rblock_sum;
//	uint32_t* d_rblock_min;
//	uint32_t* d_rblock_max;
//	CHECK(cudaMalloc((void**)&d_rblock_sum, sizeof(uint32_t) * rblock_count));
//	CHECK(cudaMalloc((void**)&d_rblock_min, sizeof(uint32_t) * rblock_count));
//	CHECK(cudaMalloc((void**)&d_rblock_max, sizeof(uint32_t) * rblock_count));
//	fc_range_summimmax<<<fc2rgrid, fc2rblock>>>(d_ranges, rblock_count, d_rblock_sum, d_rblock_min, d_rblock_max);
//
//	CHECK(cudaDeviceSynchronize());
//
//	/*
//		3.コントラストスケーリング・輝度シフト計算
//	*/
//
//	dim3 fc3block(THREADBLOCK_MAX);
//	dim3 fc3grid(dblock_count, rblock_count / THREADBLOCK_MAX);
//
//	double* d_contrast_scaling;
//	uint32_t* d_brightness_shift;
//
//	CHECK(cudaMalloc((void**)&d_contrast_scaling, sizeof(double) * dblock_count * rblock_count));
//	CHECK(cudaMalloc((void**)&d_brightness_shift, sizeof(uint32_t) * dblock_count * rblock_count));
//	//CHECK(cudaMalloc((void**)&d_adjust_domains_for_ranges, sizeof(uint32_t) * dblock_count * rblock_count * dr_block_pixel_total));
//
//	fc_calc_scale_n_shift<<<fc3grid, fc3block>>>(d_dblock_sum,
//												 d_dblock_min,
//												 d_dblock_max,
//												 d_rblock_sum,
//									     		 d_rblock_min,
//												 d_rblock_max,
//												 dr_block_pixel_total,
//											     d_contrast_scaling,
//											     d_brightness_shift);
//
//	CHECK(cudaDeviceSynchronize());
//
//	/*
//		4.コントラストスケーリング・輝度シフト適用・回転・鏡像変換・二乗計算・MSE計算（リダクション）
//	*/
//
//	//
//	dim3 fc4block(dr_blocksize, dr_blocksize, THREADBLOCK_MAX / dr_block_pixel_total);
//	dim3 fc4grid(dblock_count , rblock_count / fc4block.z);
//	//各レンジの各ドメインの各回転変換後のMSEを保存しておく
//	uint32_t* d_mse;
//	CHECK(cudaMalloc((void**)&d_mse, sizeof(uint32_t) * rblock_count * dblock_count * 8));
//
//	fc_transform_n_calc_mse<<<fc4grid, fc4block>>>(d_domains,
//												   d_ranges,
//												   d_contrast_scaling,
//												   d_brightness_shift,
//												   d_mse);
//	CHECK(cudaDeviceSynchronize());
//	//std::cout << sizeof(uint32_t) * rblock_count * dblock_count * 8 << "byte" << std::endl;
//
//	//uint32_t* h_mse = new uint32_t[rblock_count * dblock_count * 8];
//	//CHECK(cudaMemcpy(h_mse, d_mse, sizeof(uint32_t) * rblock_count * dblock_count * 8, cudaMemcpyDeviceToHost));
//
//	//std::cout << "nukiuti : " << h_mse[231] << std::endl;
//
//	//delete[] h_mse;
//
//	/*
//		5.各レンジ毎最小MSE・index計算(リダクション)
//	*/
//	//ブロックサイズに応じたコンスタントメモリを初期化する
//	init_affine_transformer(dr_blocksize);
//	dim3 fc5block(THREADBLOCK_MAX);
//	dim3 fc5grid(dblock_count * 8, rblock_count/ THREADBLOCK_MAX);
//
//	//レンジの数だけ圧縮データを保存する
//	compress_data_part_rgb_gpu* h_compress_data = new compress_data_part_rgb_gpu[rblock_count];
//	compress_data_part_rgb_gpu* d_compress_data;
//	CHECK(cudaMalloc((void**)&d_compress_data, sizeof(compress_data_part_rgb_gpu) * rblock_count));
//
//	fc_save_min_mse<<<fc5grid, fc5block>>>(d_mse, d_contrast_scaling, d_brightness_shift, d_compress_data);
//	
//	CHECK(cudaMemcpy(h_compress_data, d_compress_data, sizeof(compress_data_part_rgb_gpu) * rblock_count, cudaMemcpyDeviceToHost));
//
//	//delete[] h_ranges;
//	//delete[] h_domains;
//
//	/*
//		L.後処理
//	*/
//
//	delete[] h_orig_img;
//	delete[] h_compress_data;
//
//	CHECK(cudaFree(d_orig_img));
//	CHECK(cudaFree(d_ranges));
//	CHECK(cudaFree(d_domains));
//
//	CHECK(cudaFree(d_rblock_sum));
//	CHECK(cudaFree(d_rblock_min));
//	CHECK(cudaFree(d_rblock_max));
//	
//	CHECK(cudaFree(d_dblock_sum));
//	CHECK(cudaFree(d_dblock_min));
//	CHECK(cudaFree(d_dblock_max));
//
//	CHECK(cudaFree(d_contrast_scaling));
//	CHECK(cudaFree(d_brightness_shift));
//
//	CHECK(cudaFree(d_mse));
//
//	CHECK(cudaDeviceSynchronize());
//}
