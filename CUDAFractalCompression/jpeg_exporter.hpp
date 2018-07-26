/*
	jpeg圧縮を行いファイルに書き出す
	内部的にはopencv内のlibjepgに頼るだけである
*/

#pragma once

#include <opencv2/core.hpp>

void export_jpeg(std::string, const cv::Mat input);
