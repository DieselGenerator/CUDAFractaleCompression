#pragma once

#include <opencv2/core.hpp>

void affine_transform(const cv::Mat&, const uint8_t, cv::Mat&);
void resize_half(const cv::Mat&, cv::Mat&, const uint32_t depth = 1);
