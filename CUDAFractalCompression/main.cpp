/*
	vs2015ではctrl + F5でビルドすればassertが有効化される
*/

#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>    // std::copy
#include <iterator>     // std::back_inserter

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <cuda_runtime_api.h>

//test module
#include "dummy_kernel.cuh"
#include "arithmetic_test.cuh"
#include "transform_generator.hpp"

//main
#include "config.hpp"
#include "images.hpp"
#include "comparators.hpp"
#include "csv_util.hpp"
#include "jpeg_exporter.hpp"
#include "test.hpp"
#include "fractale_compressor.hpp"
#include "fractale_compressor_gpu.cuh"
#include "fractale_compressor_gpu_reduce_domains.cuh"
#include "fractale_compressor_gpu_reduce_ranges.cuh"
#include "fractale_decompressor.hpp"

int main(int argc, char *argv[])
{
	config_assert();

	if (config::enable_gpu_warmup) {
		kernel_ready();
	}
	if (config::enable_arithmetic_speedtest) {
		arithmetic_speedtest();
	}
	if (config::enable_transform_generator) {
		transform_generator();
	}

	Images images;

	images.loadImagesToHost();
	images.convert2greyHost();
	//images.showAllHostImages();
	images.writeGreyImages();

	FractaleCompressor fc;
	FractaleDecompressor fd;

	std::chrono::system_clock::time_point start, end;
	std::string separator(40, '-');

	/*
		各モジュールのテスト
	*/
	{
		//bit_packing_test(images);
	}

	/*
		CPU上でのOpenCVを用いたフラクタル画像圧縮，展開
	*/
	{
		ifs_header header;
		uint32_t counter = 0;
		std::vector<ifs_transformer*> ifs_data;
		//compress
		if (config::enable_cpu_opencv_compress) {

			std::cout << separator << std::endl;

			for (auto entry : images.h_grey_images) {

				header.image_height = entry.second.rows;
				header.image_width = entry.second.cols;

				std::cout << "cpu (use opencv) compressing : " << entry.first << std::endl;
				start = std::chrono::system_clock::now();

				ifs_data = fc.compress(entry.second);
				
				/*テストコード*/
				//import_csv("lena-512x512.png-mse0.csv", ifs_data);

				end = std::chrono::system_clock::now();
				long long  encode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				std::cout << "fractale compressing(CPU) elapsed " << encode_time << " milli sec \n";

				double psnr;
				long long decode_time;
				const std::string prefix("(CPU original kernel)");

				if (config::enable_cpu_opencv_decompress) {
					//decompress
					std::cout << "cpu (use opencv) decompressing : " << entry.first << std::endl;
					start = std::chrono::system_clock::now();

					cv::Mat decompressed_image = fd.decompress(header, ifs_data);

					end = std::chrono::system_clock::now();
					decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					std::cout << "fractale decompressing(CPU) elapsed " << decode_time << " milli sec \n";

					psnr = calcPSNR(entry.second, decompressed_image);
					std::cout << "psnr : " << psnr << std::endl;

					if (config::enable_cpu_opencv_output) {
						cv::imwrite("out\\" + prefix + entry.first, decompressed_image);
						std::cout << "output : out\\" + prefix + entry.first << std::endl;
					}
				}

				export_csv(prefix + entry.first, ifs_data, encode_time, decode_time, psnr);

				std::cout << ++counter << " / " << images.h_grey_images.size() << " completed" << std::endl;
				std::cout << separator << std::endl;
			}
		}
		//後始末
		for (ifs_transformer* c : ifs_data) {
			delete c;
		}
	}

	/*
		GPU上でのOpenCVを用いた画像圧縮
	*/
	{	ifs_header header;
		std::vector<ifs_transformer*> ifs_data;
		uint32_t counter = 0;

		if (config::enable_gpu_opencv_compress) {
			images.uploadImagesToDevice();
			images.convert2greyDevice();
			//(GPUでの)グレースケールの圧縮
			for (auto entry : images.d_grey_images) {
				std::cout << "GPU compressing : " << entry.first << std::endl;
				start = std::chrono::system_clock::now();

				fc.compress(entry.second);

				end = std::chrono::system_clock::now();
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				std::cout << "elapsed " << elapsed << " milli sec \n";
			}
		}
	}

	/*
		GPU上でオリジナルカーネルによるフラクタル画像圧縮
	*/
	{	
		ifs_header header;
		std::vector<ifs_transformer*> ifs_data;
		uint32_t counter = 0;

		if (config::enable_gpu_compress) {
			for (auto entry : images.h_grey_images) {

				header.image_height = entry.second.rows;
				header.image_width = entry.second.cols;

				std::cout << "GPU (original kernel) grayscale compressing : " << entry.first << std::endl;
				start = std::chrono::system_clock::now();

				ifs_data = launch_compress_kernel(entry.second, RANGE_SIZE);

				end = std::chrono::system_clock::now();
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				std::cout << "elapsed " << elapsed << " milli sec \n";

				double psnr;
				long long decode_time;
				const std::string prefix("(GPU original kernel)");

				if (config::enable_gpu_decompress) {
					std::cout << "GPU (original kernel) grayscale decompressing : " << entry.first << std::endl;
					start = std::chrono::system_clock::now();

					cv::Mat decompressed_image = fd.decompress(header, ifs_data);

					end = std::chrono::system_clock::now();
					decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					std::cout << "GPU (original kernel) grayscale decompressing elapsed " << decode_time << " milli sec \n";

					psnr = calcPSNR(entry.second, decompressed_image);
					std::cout << "psnr : " << psnr << std::endl;

					if (config::enable_gpu_output) {
						cv::imwrite("out\\" + prefix + entry.first, decompressed_image);
						std::cout << "output : out\\" + prefix + entry.first << std::endl;
						export_csv(prefix + entry.first, ifs_data, elapsed, decode_time, psnr);
					}
				}

				std::cout << ++counter << " / " << images.h_grey_images.size() << " completed" << std::endl;
				std::cout << separator << std::endl;
			}
		}
	}

	/*
		GPU上で探索ドメイン減らしたフラクタル画像圧縮
	*/
	{	
		ifs_header header;
		std::vector<ifs_transformer*> ifs_data;
		uint32_t counter = 0;

		if (config::enable_gpu_reduce_domains_compress) {
			for (auto entry : images.h_grey_images) {

				uint32_t dblock_cols = (entry.second.cols / RANGE_SIZE) >> 1;
				uint32_t dblock_rows = (entry.second.rows / RANGE_SIZE) >> 1;

				//デフォルトの並列ブロック数
				//uint32_t dblock_limit = dblock_cols * dblock_rows;
				for (uint32_t dblock_limit = dblock_cols * (dblock_rows - 1); dblock_limit > dblock_cols; dblock_limit -= dblock_cols){

					header.image_height = entry.second.rows;
					header.image_width = entry.second.cols;

					std::cout << "GPU (reduce domains kernel) grayscale compressing : " << entry.first << std::endl;
					start = std::chrono::system_clock::now();

					ifs_data = launch_reduce_domains_compress_kernel(entry.second, RANGE_SIZE, dblock_limit);
					//ifs_data = launch_reduce_ranges_compress_kernel(entry.second, RANGE_SIZE, dblock_limit);

					end = std::chrono::system_clock::now();
					auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					std::cout << "elapsed " << elapsed << " milli sec \n";

					double psnr;
					long long decode_time;
					std::string prefix = std::string("(GPU reduce domains)") + std::to_string(dblock_limit);

					if (config::enable_gpu_reduce_domains_decompress) {
						std::cout << "GPU (reduce ranges kernel) grayscale decompressing : " << entry.first << std::endl;
						start = std::chrono::system_clock::now();

						cv::Mat decompressed_image = fd.decompress(header, ifs_data);

						end = std::chrono::system_clock::now();
						decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
						std::cout << "GPU (reduce domains kernel) grayscale decompressing elapsed " << decode_time << " milli sec \n";

						psnr = calcPSNR(entry.second, decompressed_image);
						std::cout << "psnr : " << psnr << std::endl;

						if (config::enable_gpu_reduce_domains_output) {
							cv::imwrite("out\\" + prefix + entry.first, decompressed_image);
							std::cout << "output : out\\" + prefix + entry.first << std::endl;
						}
					}

					std::string s_psnr = std::to_string(psnr);
					std::string domain_count = std::to_string(dblock_limit);
					std::string time = std::to_string(elapsed);
					std::string filename = "(GPU redece domain)" + entry.first + "[psnr=" + s_psnr + ",domain_count=" + domain_count + ",time=" + time + "].jpg";

					if (config::enable_gpu_reduce_domains_output) {
						export_csv(filename, ifs_data, elapsed, decode_time, psnr);
					}

					std::cout << ++counter << " / " << images.h_grey_images.size() << " completed" << std::endl;
					std::cout << separator << std::endl;
				}
			}
		}
	}

	/*
		GPU上で探索レンジ数を減らしたフラクタル画像圧縮
	*/
	{	
		ifs_header header;
		std::vector<ifs_transformer*> ifs_data;
		std::vector<ifs_transformer*> ifs_inner_data;
		std::vector<ifs_transformer*> ifs_outer_data;
		uint32_t counter = 0;

		if (config::enable_gpu_reduce_ranges_compress) {
			for (auto entry : images.h_grey_images) {

				uint32_t dblock_cols = (entry.second.cols / RANGE_SIZE) >> 1;
				uint32_t dblock_rows = (entry.second.rows / RANGE_SIZE) >> 1;

				//外周をどれだけ大きいブロックとして処理するか
				for (uint32_t reduce_range_config = 0; reduce_range_config <= 2; reduce_range_config++){

					//内部のブロックサイズ
					uint32_t inner_blocksize = 4;
					//外部のブロックサイズ
					uint32_t outer_blocksize = 8;
					//内部部分を計算する時の外周分の大きさ(ブロック単位)
					uint32_t periphery_inner_size = 2;
					//外周部分を計算する時の外周分の大きさ(ブロック単位)
					uint32_t periphery_outer_size = 1;
	
					/*
						レンジの減らし方config部分
					*/
					switch (reduce_range_config) {
					case 0:
						inner_blocksize = 4;
						outer_blocksize = 8;
						periphery_inner_size = 2;
						periphery_outer_size = 1;
						break;
					case 1:
						inner_blocksize = 4;
						outer_blocksize = 8;
						periphery_inner_size = 4;
						periphery_outer_size = 2;
						break;
					case 2:
						inner_blocksize = 4;
						outer_blocksize = 16;
						periphery_inner_size = 4;
						periphery_outer_size = 1;
						break;
					default:
						inner_blocksize = 4;
						outer_blocksize = 8;
						periphery_inner_size = 2;
						periphery_outer_size = 1;
					}

					header.image_height = entry.second.rows;
					header.image_width = entry.second.cols;

					std::cout << "GPU (reduce domains kernel) grayscale compressing : " << entry.first << std::endl;
					start = std::chrono::system_clock::now();

					ifs_inner_data = launch_reduce_ranges_compress_kernel(entry.second, inner_blocksize, true, periphery_inner_size);
					ifs_outer_data = launch_reduce_ranges_compress_kernel(entry.second, outer_blocksize, false, periphery_outer_size);

					cv::waitKey();

					end = std::chrono::system_clock::now();
					auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					std::cout << "elapsed " << elapsed << " milli sec \n";

					/*
						2つのベクタを1つのベクタに統合する
					*/
					ifs_data.clear();
					std::copy(ifs_inner_data.begin(), ifs_inner_data.end(), std::back_inserter(ifs_data));
					ifs_data.insert(ifs_data.end(), ifs_outer_data.begin(), ifs_outer_data.end());

					double psnr;
					long long decode_time;
					std::string prefix = std::string("(GPU reduce ranges) range config ") + std::to_string(reduce_range_config);

					if (config::enable_gpu_reduce_ranges_decompress) {
						std::cout << "GPU (reduce ranges kernel) grayscale decompressing : " << entry.first << std::endl;
						start = std::chrono::system_clock::now();

						cv::Mat decompressed_image = fd.decompress(header, ifs_data);

						end = std::chrono::system_clock::now();
						decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
						std::cout << "GPU (reduce domains kernel) grayscale decompressing elapsed " << decode_time << " milli sec \n";

						psnr = calcPSNR(entry.second, decompressed_image);
						std::cout << "psnr : " << psnr << std::endl;

						if (config::enable_gpu_reduce_ranges_output) {
							cv::imwrite("out\\" + prefix + entry.first, decompressed_image);
							std::cout << "output : out\\" + prefix + entry.first << std::endl;
						}
					}

					if (config::enable_gpu_reduce_ranges_output) {
						export_csv(prefix + entry.first, ifs_data, elapsed, decode_time, psnr);
					}

					std::cout << ++counter << " / " << images.h_grey_images.size() << " completed" << std::endl;
					std::cout << separator << std::endl;
				}
			}
		}
	}

	/*
		CPU上でopenCV内部のlibjpegを用いたJPEG圧縮
	*/
	if (config::enable_jpeg_compress) {
		for (auto entry : images.h_grey_images) {

			std::string statistics_filename = "out\\" + entry.first + "-statistics.csv";
			std::string index = "psnr[dB],quality,time[ms],size[bytes]";
			std::ofstream ofs;
			std::string statistics_data;

			if (config::enable_jpeg_repeat && config::enable_jpeg_output_statistics) {
				ofs.open(statistics_filename);
				ofs << index << std::endl;
			}

			int quality = config::enable_jpeg_repeat ? 100 : 95;

			do {

				//export_jpeg(entry.first, entry.second);

				std::cout << "JPEG compressing : " << entry.first << std::endl;
				start = std::chrono::system_clock::now();

				std::vector<uchar> buff;//buffer for coding
				std::vector<int> param = std::vector<int>(2);
				param[0] = CV_IMWRITE_JPEG_QUALITY;
				param[1] = quality;//default(95) 0-100

				cv::imencode(".jpg", entry.second, buff, param);

				end = std::chrono::system_clock::now();

				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				std::cout << "JPEG compressing elapsed " << elapsed << " milli sec \n";

				std::cout << "JPEG decompressing : " << entry.first << std::endl;
				start = std::chrono::system_clock::now();

				cv::Mat jpegimage = cv::imdecode(cv::Mat(buff), CV_LOAD_IMAGE_UNCHANGED);

				end = std::chrono::system_clock::now();

				elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				std::cout << "JPEG decompressing elapsed " << elapsed << " milli sec \n";

				std::string psnr = std::to_string(calcPSNR(entry.second, jpegimage));
				std::string s_quality = std::to_string(param[1]);
				std::string time = std::to_string(elapsed);
				std::string filename = "out\\(CPU-JPEG)" + entry.first + "[psnr=" + psnr + ",quality=" + s_quality + ",time=" + time + "].jpg";
				
				if (config::enable_jpeg_output) {
					cv::imwrite(filename, jpegimage);
				}

			// for ifstream
				std::ifstream ifs(filename, std::ios::binary);
				ifs.seekg(0, std::ios::end);
				auto eofpos = ifs.tellg();
				ifs.clear();
				ifs.seekg(0, std::ios::beg);
				auto begpos = ifs.tellg();
				auto size2 = eofpos - begpos;
				ifs.close();	

				quality--;
				std::string data = psnr + "," + s_quality + "," + time + "," + std::to_string(size2);
				
				if (config::enable_jpeg_repeat && config::enable_jpeg_output_statistics) {
					ofs << data << std::endl;
				}

			} while (config::enable_jpeg_repeat && quality > 0);

			if (config::enable_jpeg_repeat && config::enable_jpeg_output_statistics) {
				ofs.close();
			}

		}
	}

	//
	std::cout << "press Key to quit..." << std::endl;
	getchar();

	return 0;
}
