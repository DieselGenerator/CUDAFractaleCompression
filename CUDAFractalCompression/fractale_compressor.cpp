#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <boost/range/algorithm_ext/erase.hpp>

#include "fractale_compressor.hpp"
#include "affine_transformer.hpp"
#include "compress_data.hpp"
#include "ifs_transform_data.hpp"
#include "config.hpp"

//コンストラクタ
FractaleCompressor::FractaleCompressor() {

}

uint8_t FractaleCompressor::contrast_scaling(const cv::Mat& domain, const cv::Mat& range) {
	double d_min, d_max;
	cv::minMaxLoc(domain, &d_min, &d_max);
	double r_min, r_max;
	cv::minMaxLoc(range, &r_min, &r_max);
	double scaling_factor = ((r_max - r_min) / (d_max - d_min));
	
	uint8_t scaling_save = 0;

	//情報を落とす
	for(double i = 0.0625; i <= 1; i += 0.0625){
		if( (i - 0.0625) < scaling_factor && scaling_factor < i){
			return scaling_save;
		}
		scaling_save++;
	}
	return 0xF;
}

uint8_t FractaleCompressor::brightness_shift(const cv::Mat& domain, const cv::Mat& range) {
	double d_sum = cv::sum(domain)[0];
	double r_sum = cv::sum(range)[0];
	double shift_factor = (r_sum - d_sum) / (range.cols * range.rows);
	if (shift_factor  < 0 ){
		return 0;
	}
	if (shift_factor > 255){
		return 255;
	}
	return (uint8_t)shift_factor;
}

void  FractaleCompressor::ifs_init(uint32_t rblock_x, uint32_t rlobck_y, uint8_t rsize, ifs_transformer* ifs){
	ifs->error = std::numeric_limits<double>::max();
	ifs->rblock_x = rblock_x;
	ifs->rblock_y = rlobck_y;
	ifs->blocksize = rsize;
}

/*
input:
	cv::Mat domains		:	縮小された画像データ(ドメイン全体を使う)
	cv::Mat range		:	レンジブロック１つ
output:
	compressed_data* c	:	
details:
	レンジブロック１つに対して最適なifs変換を計算する

*/
void FractaleCompressor::ifs_transform(const cv::Mat domains, const cv::Mat range, ifs_transformer* c) {

	const int32_t range_size = range.cols;// == range.rows
	//初期化は最大
	double best_mse = std::numeric_limits<double>::max();

	for (int domain_y = 0; domain_y < domains.rows; domain_y = domain_y + range_size) {
		for (int domain_x = 0; domain_x < domains.cols; domain_x = domain_x + range_size) {
			//std::cout << c->dblock_id << std::endl;
			
			//ドメインブロック1つ
			cv::Mat domain(domains, cv::Rect(domain_x, domain_y, range_size, range_size));
			
			//レンジブロック，ドメインブロック間のブライトネスシフト及びコントラストスケーリングを求める
			//contrast, brightness
			cv::Mat fixed_domain = domain.clone();
			uint8_t scaling_factor = contrast_scaling(domain, range);
			uint8_t shift_factor = brightness_shift(domain, range);

			//コントラストスケーリング及びブライトネスシフトを適用する
			fixed_domain = fixed_domain * ((scaling_factor + 1) * 0.0625) + shift_factor;

			for (uint8_t affine_pattern = 0; affine_pattern < 8; affine_pattern++) {
				cv::Mat derived_domain;
				affine_transform(fixed_domain, affine_pattern, derived_domain);
				double mse = calcMSE(range, derived_domain);
				if (mse < best_mse){
					c->dblock_x = domain_x;
					c->dblock_y = domain_y;
					c->affine = affine_pattern;
					c->scaling = scaling_factor;
					c->shift = shift_factor;
					c->error = mse;
					best_mse = mse;
					//std::cout << best_mse << std::endl;
				}
				//全探索をしない場合mseがthreshold以下なら探索を終了する
				if(best_mse < config::mse_threshold){
					return;
				}
			}
		}
	}

}

/*
input:	
	cv::Mat input				:	フラクタル圧縮を行う画像
return:	
	std::vector<ifs_transformer>:	ifs変換の集合

	フラクタル圧縮の実体，CPU(OpenCV使用)版
*/
std::vector<ifs_transformer*> FractaleCompressor::compress(cv::Mat input) {
	
	const uint32_t width = input.cols;
	const uint32_t height = input.rows;
	
	assert((width % config::domain_size_max) == 0);

	std::cout << "image size : width " << input.cols << " height " << input.rows << std::endl;

	//ifs_header* header = new ifs_header;
	std::vector<ifs_transformer*> ifs_data;

	//複数チャンネル
	if (input.channels() > 1){
		//未実装
		std::cout << "typea: " << CV_8UC3 << std::endl;
		std::cout << "type : " << input.type() << std::endl;
		cv::waitKey();

		return ifs_data;
	}

	//画素値は0-255の範囲のみとる
	cv::Mat input_8U;
	input.convertTo(input_8U, CV_8U);

	//四分岐分割の準備
	//最初のみ全てのレンジに対して最適なifs変換を求める
	{
		uint8_t rsize = config::range_size_max;
		//入力の二分の一サイズのドメイン
		cv::Mat domains;
		cv::resize(input_8U, domains, cv::Size(width >> 1, height >> 1), cv::INTER_NEAREST);
		for (uint32_t range_y = 0; range_y < height; range_y = range_y + rsize) {
			for (uint32_t range_x = 0; range_x < width; range_x = range_x + rsize) {
				//レンジを切り抜く
				cv::Mat range(input_8U, cv::Rect(range_x, range_y, rsize, rsize));
				ifs_transformer* c = new ifs_transformer();
				c->error = std::numeric_limits<double>::max();
				c->rblock_x = range_x;
				c->rblock_y = range_y;
				c->blocksize = rsize;
				ifs_transform(domains, range, c);
				ifs_data.push_back(c);
			}
		}
	}

	//四分岐分割圧縮を行う
	if(config::enable_quartree_compress){
		//四分岐分割ではドメインの大きさは2^depth分の1になる
		uint8_t depth = 1/*2*/;
		uint32_t stage = 0;
		for (uint8_t rsize = config::range_size_max >> 1; rsize >= config::range_size_min; rsize >>= 1/*, depth++*/) {
			std::vector<ifs_transformer*> ifs_quadtree;
			cv::Mat domains;
			cv::resize(input_8U, domains, cv::Size(width >> depth, height >> depth), cv::INTER_NEAREST);
			//
			uint32_t counter = 0;
			stage++;
			for(ifs_transformer* ifs : ifs_data){
				//閾値に満たない領域は開放，vectorに入ったままの無意味なポインタは後でまとめてvectorから削除する
				if (ifs->error > config::mse_threshold){

					ifs_transformer* c0 = new ifs_transformer();
					ifs_init(ifs->rblock_x, ifs->rblock_y, rsize, c0);
					cv::Mat range0(input_8U, cv::Rect(c0->rblock_x, c0->rblock_y, rsize, rsize));
					ifs_transform(domains, range0, c0);
					ifs_quadtree.push_back(c0);

					ifs_transformer* c1 = new ifs_transformer();
					ifs_init(ifs->rblock_x + rsize, ifs->rblock_y, rsize, c1);
					cv::Mat range1(input_8U, cv::Rect(c1->rblock_x, c1->rblock_y, rsize, rsize));
					ifs_transform(domains, range1, c1);
					ifs_quadtree.push_back(c1);

					ifs_transformer* c2 = new ifs_transformer();
					ifs_init(ifs->rblock_x, ifs->rblock_y + rsize, rsize, c2);
					cv::Mat range2(input_8U, cv::Rect(c2->rblock_x, c2->rblock_y, rsize, rsize));
					ifs_transform(domains, range2, c2);
					ifs_quadtree.push_back(c2);

					ifs_transformer* c3 = new ifs_transformer();
					ifs_init(ifs->rblock_x + rsize, ifs->rblock_y + rsize, rsize, c3);
					cv::Mat range3(input_8U, cv::Rect(c3->rblock_x, c3->rblock_y, rsize, rsize));
					ifs_transform(domains, range3, c3);
					ifs_quadtree.push_back(c3);

					std::cout << "stage" << stage << " ifs divied :" << ++counter << "/" << ifs_data.size() << "\r" << std::flush;
				}

			}
			std::cout << std::endl;

			//std::cout << "size1:" << ifs_data.size() << std::endl;
			boost::remove_erase_if(ifs_data, [](ifs_transformer* ifs) { bool r = ifs->error > config::mse_threshold; if (r) { delete ifs; } return r; });
			//std::cout << "size2:" << ifs_data.size() << std::endl;
			std::copy(ifs_quadtree.begin(), ifs_quadtree.end(), std::back_inserter(ifs_data));
			//std::cout << "size3:" << ifs_data.size() << std::endl;
		}
	}
	
	//uint8_t depth = 3;

	////4分割木
	//for (uint8_t rsize = config::range_size_max; rsize >= config::range_size_min; rsize <<= 1, depth++){

	//	std::cout << "range size : " << (uint32_t)rsize << std::endl;

	//	//縮小変換したドメインを用意しておく
	//	cv::Size domain_cvsize = cv::Size(width >> depth, height >> depth);
	//	cv::Mat domain_resize;
	//	cv::resize(input_8U, domain_resize, domain_cvsize, cv::INTER_NEAREST);

	//	for (uint32_t range_y = 0; range_y < height; range_y = range_y + rsize) {
	//		for (uint32_t range_x = 0; range_x < width; range_x = range_x + rsize) {
	//			//レンジを切り抜く
	//			cv::Rect range_rect(range_x, range_y, rsize, rsize);
	//			cv::Mat range(input_8U, range_rect);
	//			ifs_transformer* c = new ifs_transformer();
	//			c->error = std::numeric_limits<double>::max();
	//			c->rblock_x = range_x;
	//			c->rblock_y = range_y;
	//			c->blocksize = rsize;
	//			//各レンジについて最適なifs変換を求める
	//			ifs_transform(domain_resize, range, c);
	//			ifs_data.push_back(c);
	//			//std::cout << "dx : " << c->dblock_x << "dy : " << c->dblock_y << std::endl;
	//		}
	//	}
	//	//TODO
	//	break;

	//}

	return ifs_data;
}

std::vector<compressed_data> FractaleCompressor::compress(cv::cuda::GpuMat d_input) {

	cv::Size range_size = cv::Size(RANGE_SIZE, RANGE_SIZE);

	uint32_t domainCount = countDomain(d_input);
	//domainの数だけvectorの要素数を確保する

	//std::cout << return_data.size() << std::endl;
	std::vector<compressed_data> return_data(domainCount);
	std::vector<cv::cuda::GpuMat> domains(domainCount);
	std::vector<cv::cuda::GpuMat> domains_resized(domainCount);

	uint32_t counter = 0;
	for (int domain_x = 0; domain_x < d_input.cols; domain_x = domain_x + DOMAIN_SIZE) {
		for (int domain_y = 0; domain_y < d_input.rows; domain_y = domain_y + DOMAIN_SIZE) {
			cv::Rect domain_rect(domain_x, domain_y, DOMAIN_SIZE, DOMAIN_SIZE);
			//domein_rectにより切り抜かれる画像
			domains[counter] = d_input(domain_rect);
			cv::cuda::resize(domains[counter], domains_resized[counter], range_size, cv::INTER_NEAREST);
			counter++;
		}
	}

	////平均二乗誤差(初期値は∞)
	//double best_mse = std::numeric_limits<double>::max();

	////各ドメインに対しての処理
	////DOMAIN_SIZE x DOMAIN_SIZEの切り抜くための領域を用意する
	//cv::Rect domain_rect(domain_x, domain_y, DOMAIN_SIZE, DOMAIN_SIZE);
	////domein_rectにより切り抜かれる画像
	//cv::cuda::GpuMat domain_part(d_input, domain_rect);
	//cv::Mat domain_part_resize;
	////近傍を参照する方式なので2x2の平均化である
	//cv::resize(domain_part, domain_part_resize, cv::Size(RANGE_SIZE, RANGE_SIZE), cv::INTER_NEAREST);
	////2種類の鏡像変換
	//for (int mirror = 0; mirror < 2; mirror++) {
	//	//4種類の回転変換
	//	for (int rotate = 0; rotate < 4; rotate++) {
	//		//各レンジと比較
	//		for (int range_x = 0; range_x < d_input.cols; range_x = range_x + RANGE_SIZE) {
	//			for (int range_y = 0; range_y < d_input.rows; range_y = range_y + RANGE_SIZE) {
	//				//各レンジを切り出す
	//				cv::Rect range_rect(range_x, range_y, RANGE_SIZE, RANGE_SIZE);
	//				cv::Mat range_part(d_input, range_rect);
	//				//切り出したレンジとドメインの平均二乗誤差を計算する
	//				cv::Mat mse_mat = domain_part_resize - range_part;
	//				cv::Scalar s = cv::sum(mse_mat.mul(mse_mat));
	//				double new_mse = s[0] / mse_mat.rows / mse_mat.cols;
	//				if (best_mse > new_mse) {
	//					best_mse = new_mse;
	//					return_data[domain_counter].domain_x = domain_x;
	//					return_data[domain_counter].domain_y = domain_y;
	//					return_data[domain_counter].range_x = range_x;
	//					return_data[domain_counter].range_y = range_y;
	//					return_data[domain_counter].mirror = mirror;
	//					return_data[domain_counter].rotate = rotate;
	//					//std::cout << "Now mse updated domain x : " << domain_x << " domain y : " << domain_y << " mse : " << mse <<
	//					//"range x : " << range_x << " range y : " << range_y << std::endl;
	//					//閾値を設定する場合
	//					//if (mse < 10) {
	//					//	goto NEXT_DOMAIN ;
	//					//}
	//				}

	//			}
	//		}
	//	NEXT_DOMAIN_TRANSFORM:;
	//	}
	//}
	//domain_counter++;
	////cv::namedWindow("test", CV_WINDOW_AUTOSIZE);
	////cv::imshow("test", domain32);
	////cv::waitKey(0);



//return_data.
//countDomain(d_input);




	return return_data;
}

template <typename MAT>
uint32_t FractaleCompressor::countDomain(MAT mat) {
	if (mat.rows == -1 || mat.cols == -1) {
		std::cout << "error in countDomain()" << std::endl;
	}
	if (mat.rows % DOMAIN_SIZE != 0 || mat.cols % DOMAIN_SIZE != 0) {
		std::cout << "image size must be (N * " << DOMAIN_SIZE << ") x (M *" << DOMAIN_SIZE << ")" << std::endl;
		std::cout << "compress will be failed" << std::endl;
	}
	std::cout << mat.rows << " : rows" << std::endl;
	return (mat.rows / DOMAIN_SIZE) * (mat.cols / DOMAIN_SIZE);
}

//平均二乗誤差を計算する
double FractaleCompressor::calcMSE(const cv::Mat& gray1, const cv::Mat& gray2) {
	cv::Mat gray16s1, gray16s2;

	gray1.convertTo(gray16s1, CV_16S);
	gray2.convertTo(gray16s2, CV_16S);

	gray16s1 -= gray16s2;
	cv::Scalar s = sum(gray16s1.mul(gray16s1));
	return (s[0] / gray16s1.rows / gray16s1.cols);
}

//平均二乗誤差を計算する
double FractaleCompressor::calcMSEoptimized(cv::cuda::GpuMat d_g1, cv::cuda::GpuMat d_g2, bufferMSE& buf) {

	d_g1.convertTo(buf.d_t1, CV_32F);
	d_g2.convertTo(buf.d_t2, CV_32F);

	//std::cout << buf.d_t1.reshape(1).size() << ":" << buf.d_t2.reshape(1).size() << std::endl;
	//std::cout << buf.d_t1.reshape(1).type() << ":" << buf.d_t2.reshape(1).type() << std::endl;

	cv::cuda::absdiff(buf.d_t1.reshape(1), buf.d_t2.reshape(1), buf.d_gs);
	cv::cuda::multiply(buf.d_gs, buf.d_gs, buf.d_gs);

	cv::Scalar sum = cv::cuda::sum(buf.d_gs, buf.d_buf)[0];
	double sse = sum.val[0] + sum.val[1] + sum.val[2];

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double mse = sse / (double)(d_g1.channels() * d_g1.rows * d_g1.cols);
		//double psnr = 10.0*log10((255 * 255) / mse);
		return mse;
	}
}

//std::vector<compressed_data> return_data;
//bufferMSE buf;
////cv::cuda::GpuMat i1, i2;
////domainの数だけvectorの要素数を確保する
//return_data.resize(countDomain(input));
//std::cout << return_data.size() << std::endl;
//uint32_t domain_counter = 0;
//cv::Size range_cvsize = cv::Size(RANGE_SIZE, RANGE_SIZE);

//for (int domain_x = 0; domain_x < input.cols; domain_x = domain_x + DOMAIN_SIZE) {
//	for (int domain_y = 0; domain_y < input.rows; domain_y = domain_y + DOMAIN_SIZE) {

//		//平均二乗誤差(初期値は∞)
//		double best_mse = std::numeric_limits<double>::max();

//		//各ドメインに対しての処理
//		//DOMAIN_SIZE x DOMAIN_SIZEの切り抜くための領域を用意する
//		cv::Rect domain_rect(domain_x, domain_y, DOMAIN_SIZE, DOMAIN_SIZE);
//		//domein_rectにより切り抜かれる画像
//		cv::Mat domain_part(input, domain_rect);
//		cv::Mat domain_part_resize;
//		//近傍を参照する方式なので2x2の平均化である
//		cv::resize(domain_part, domain_part_resize, range_cvsize, cv::INTER_NEAREST);
//		//2種類の鏡像変換
//		for (int mirror = 0; mirror < 2; mirror++) {
//			if (mirror) {
//				cv::flip(domain_part_resize, domain_part_resize, 1);;
//			}
//			//4種類の回転変換
//			for (int rotate = 0; rotate < 4; rotate++) {
//				if (rotate > 0) {
//					cv::transpose(domain_part_resize, domain_part_resize);
//					cv::flip(domain_part_resize, domain_part_resize, 1);
//				}
//				//各レンジと比較
//				for (int range_x = 0; range_x < input.cols; range_x = range_x + RANGE_SIZE) {
//					for (int range_y = 0; range_y < input.rows; range_y = range_y + RANGE_SIZE) {
//						//各レンジを切り出す
//						cv::Rect range_rect(range_x, range_y, RANGE_SIZE, RANGE_SIZE);
//						cv::Mat range_part(input, range_rect);
//						//切り出したレンジとドメインの平均二乗誤差を計算する
//						//cv::Mat mse_mat = domain_part_resize - range_part;
//						//cv::Scalar s = cv::sum(mse_mat.mul(mse_mat));
//						//double new_mse = s[0] / mse_mat.rows / mse_mat.cols;

//						double new_mse = calcMSE(domain_part_resize, range_part);


//						//i1.upload(domain_part_resize);
//						//i2.upload(range_part);
//						//double new_mse = calcMSEoptimized(i1, i2, buf);

//						if (best_mse > new_mse) {
//							best_mse = new_mse;
//							return_data[domain_counter].dblock_id = DOMAIN_SIZE * domain_y + domain_x;
//							return_data[domain_counter].rblock_id = RANGE_SIZE * range_y + range_x;
//							return_data[domain_counter].mirror = (bool)mirror;
//							return_data[domain_counter].rotate = (uint8_t)rotate;
//							//std::cout << "Now mse updated domain x : " << domain_x << " domain y : " << domain_y << " mse : " << mse <<
//								//"range x : " << range_x << " range y : " << range_y << std::endl;
//							//閾値を設定する場合
//							//if (mse < 10) {
//							//	goto NEXT_DOMAIN ;
//							//}
//						}

//					}
//				}
//			NEXT_DOMAIN_TRANSFORM:;
//			}
//		}
//		domain_counter++;
//		//if (domain_counter % 1000 == 0 ){
//		std::cout << domain_counter << std::endl;
//		//}
//		//cv::namedWindow("test", CV_WINDOW_AUTOSIZE);
//		//cv::imshow("test", domain32);
//		//cv::waitKey(0);
//	}
//}
////getchar();
//return return_data;
