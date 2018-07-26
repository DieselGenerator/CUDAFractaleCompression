#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "affine_transformer.hpp"

/*
input:
	const cv::Mat& domain		:	幾何変換を適用するドメインブロック
	const uint8_t affine_pattern:	アフィン変換のパターンを指定する
output:
	cv::Mat& derived_domain		:	幾何変換を適用したドメインブロックを保存する
*/
void affine_transform(
	const cv::Mat& domain,
	const uint8_t affine_pattern,
	cv::Mat& derived_domain) {

	//派生したドメインブロックを保存する
	derived_domain = domain.clone();

	//鏡像変換
	if (0b100 & affine_pattern) {
		cv::flip(domain, derived_domain, 1);
	}
	
	//90度回転
	if ((0b011 & affine_pattern) == 0b001) {
		cv::transpose(derived_domain, derived_domain);
		cv::flip(derived_domain, derived_domain, 1);
		//std::cout << "90" << std::endl;
	}
	//180度回転
	else if ((0b011 & affine_pattern) == 0b010) {
		cv::flip(derived_domain, derived_domain, -1);
		//std::cout << "180" << std::endl;
	}
	//270度回転
	else if ((0b011 & affine_pattern) == 0b011) {
		cv::transpose(derived_domain, derived_domain);
		cv::flip(derived_domain, derived_domain, 0);
		//std::cout << "270" << std::endl;
	};
	/*debug用*/
	//std::cout << "pattern" << (uint32_t)affine_pattern << std::endl;
	//cv::namedWindow("Image", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	//cv::imshow("Image", derived_domain);
	//cv::waitKey(0);
}

/*
input:
	const cv::Mat& input		:	入力Mat
	const uint32_t depth		:	何倍縮小するか1で1/2倍, 2で1/4倍縮小
output:
	cv::Mat& half		:	入力Matを縮小して出力する
*/
void resize_half(
	const cv::Mat& input,
	cv::Mat& half,
	const uint32_t depth) {

	cv::Size resize_cvsize = cv::Size(input.cols >> depth, input.rows >> depth);
	cv::resize(input, half, resize_cvsize, cv::INTER_NEAREST);
}
