/*
	画像データ
*/

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <boost/filesystem.hpp>

class Images {

private :
	//画像保存フォルダ
	std::string prefix = "resource\\";
	//グレースケールの出力位置
	std::string grey_prefix = "grey\\";
public:
	//画像データを保持する
	//ホスト側
	std::map<std::string, cv::Mat> h_images;
	//デバイス側
	std::map<std::string, cv::cuda::GpuMat> d_images;

	//グレースケールの画像も保持しておく
	//ホスト側
	std::map<std::string, cv::Mat> h_grey_images;
	//デバイス側
	std::map<std::string, cv::cuda::GpuMat> d_grey_images;

	/*---------------------------------------------------------*/

	//コンストラクタ
	Images();

private:
	void getImageNames(std::vector<std::string> &image_names);

public:
	//void printImageList();

	void loadImagesToHost();
	void uploadImagesToDevice();

	void showAllHostImages();
	void showAllDeviceImages();

	void convert2greyHost();
	void convert2greyDevice();

	void writeGreyImages();
	
};
