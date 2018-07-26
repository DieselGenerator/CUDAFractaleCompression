/*
	Mat2‚Â‚ÌŠÔ‚Ì‹——£‚ğ‹‚ß‚é
*/
#pragma once

#include <opencv2/core.hpp>

double calcPSNR(const cv::Mat&, const cv::Mat&);
double calcMSE(const cv::Mat&, const cv::Mat&);
