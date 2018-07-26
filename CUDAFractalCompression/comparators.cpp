#include "comparators.hpp"

#include <opencv2/core.hpp>

double calcPSNR(const cv::Mat& i1, const cv::Mat& i2) {
	cv::Mat s1;
	cv::absdiff(i1, i2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	cv::Scalar s = sum(s1);         // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) {
		return 0;
	} else {
		double mse = sse / (double)(i1.channels() * i1.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}

double calcMSE(const cv::Mat& i1, const cv::Mat& i2) {
	cv::Mat s1, s2;

	i1.convertTo(s1, CV_16S);
	i2.convertTo(s2, CV_16S);

	s1 -= s2;
	cv::Scalar s = sum(s1.mul(s1));
	return (s[0] / s1.rows / s1.cols);
}
