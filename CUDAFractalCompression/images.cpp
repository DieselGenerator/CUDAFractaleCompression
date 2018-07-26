//プログラムで扱う画像データを扱う
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <boost/filesystem.hpp>

#include "images.hpp"

//コンストラクタ
//将来的にここで，時間計測に関係ない処理を行う
Images::Images() {

}

//引数のimageNamesにファイル名を追加していく
void Images::getImageNames(std::vector<std::string> &imageNames) {
	// カレントディレクトリ以下のファイル名を取得する
	for (std::tr2::sys::directory_iterator it("resource"), end; it != end; ++it) {
		if (it->path().filename().string().find(".png") != std::string::npos || 
			it->path().filename().string().find(".jpg") != std::string::npos ||
			it->path().filename().string().find(".bmp") != std::string::npos) {
			imageNames.push_back(it->path().filename().string());
			std::cout << "load : " << it->path().filename().string() << std::endl;
		}
	}
}

//void ImageData::printImageList() {
//	for (auto v: imageNames) {
//		std::cout << v << std::endl;
//	}
//}

//画像データを読み込む
void Images::loadImagesToHost() {
	//画像データの名前を保持する
	std::vector<std::string> imageNames;
	Images::getImageNames(imageNames);

	std::cout << "loadImagesToHost() images : " << imageNames.size() << std::endl;
	auto start = std::chrono::system_clock::now();
	uint32_t image_count = 0;
	for (int i = 0; i < (int)imageNames.size(); i++) {
		h_images[imageNames[i]] = cv::imread(prefix + imageNames[i], cv::IMREAD_UNCHANGED);
		if (h_images[imageNames[i]].empty()) {
			std::cout << "failed to load image : " << imageNames[i] << std::endl;
		}
		else {
			image_count++;
		}
	}
	auto end = std::chrono::system_clock::now();
	auto diff = end - start;
	std::cout << image_count << " image loaded." << std::endl;
	std::cout << "load image(file -> host) time = " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " msec." << std::endl;
}

//loadImagesToHostを先に呼んでおかないとなにもロードしない
void Images::uploadImagesToDevice() {
	std::cout << "loadImagesToHost() images : " << h_images.size() << std::endl;
	int count = 0;
	auto start = std::chrono::system_clock::now();
	for (auto k : h_images) {
		count++;
		(d_images[k.first]).upload(k.second);
	}
	auto end = std::chrono::system_clock::now();
	auto diff = end - start;
	std::cout << "upload image (host -> device) time = " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " msec." << std::endl;
	if (count == 0) {
		return;
	}
	std::cout << "average time image upload time = " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / count << " msec." << std::endl;
}

void Images::showAllHostImages() {
	std::cout << "showAllHostImages() images : " << h_images.size() << std::endl;
	for (auto v : h_images) {
		cv::namedWindow(v.first, cv::WINDOW_AUTOSIZE);
		cv::imshow(v.first, v.second);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}

void Images::showAllDeviceImages() {
	/*
		OpenGL付きでOpenCVビルドしとけばよかったで候
	*/
}

void Images::convert2greyHost(){
	std::cout << "makeGreyHostImages()" << std::endl;
	for (auto h_image : h_images) {
		if (h_image.second.channels() != 3) {
			if (h_image.second.channels() == 1) {
				h_grey_images[h_image.first] = h_image.second.clone();
			}
			else if (h_image.second.channels() == 3) {
				cv::cvtColor(h_image.second, h_grey_images[h_image.first], cv::COLOR_BGR2GRAY);
				h_grey_images[h_image.first].convertTo(h_grey_images[h_image.first], CV_8U);
			}
			continue;
		}

	}
}

void Images::convert2greyDevice() {
	std::cout << "makeGreyDeviceImages()" << std::endl;
	int i = 0;
	for (auto d_image : d_images) {
		cv::cuda::cvtColor(d_image.second, d_grey_images[d_image.first], cv::COLOR_BGR2GRAY);
	}
}

void Images::writeGreyImages(){
	boost::filesystem::path dir("grey");
	if (!(boost::filesystem::exists(dir))) {
		boost::filesystem::create_directory(dir);
	}
	for(auto h_grey_image : h_grey_images){
		cv::imwrite(grey_prefix + h_grey_image.first, h_grey_image.second);
	}
}
