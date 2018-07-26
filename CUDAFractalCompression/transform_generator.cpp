#include <iostream>
#include <numeric>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "transform_generator.hpp"

void rotate_90n(cv::Mat const &src, cv::Mat &dst, int angle){

	CV_Assert(angle % 90 == 0 && angle <= 360 && angle >= -360);
	if (angle == 270 || angle == -90) {
		cv::transpose(src, dst);
		cv::flip(dst, dst, 0);
	}
	else if (angle == 180 || angle == -180) {
		cv::flip(src, dst, -1);
	}
	else if (angle == 90 || angle == -270) {
		cv::transpose(src, dst);
		cv::flip(dst, dst, 1);
	}
	else if (angle == 360 || angle == 0 || angle == -360) {
		if (src.data != dst.data) {
			src.copyTo(dst);
		}
	}
}

void transform_generator(){

	uint16_t a[SIZE * SIZE];
	for(int i = 0; i < SIZE*SIZE; i++){
		a[i] = i;
	}

	cv::Mat original = cv::Mat(SIZE, SIZE, CV_16U, a);
	cv::Mat mirror;
	cv::Mat temp;

	std::cout << "--original" << std::endl;
	std::cout << original << std::endl;

	std::cout << "--original90" << std::endl;
	rotate_90n(original, temp, 90);
	std::cout << temp << std::endl;

	std::cout << "--original180" << std::endl;
	rotate_90n(original, temp, 180);
	std::cout << temp << std::endl;

	std::cout << "--original270" << std::endl;
	rotate_90n(original, temp, 270);
	std::cout << temp << std::endl;

	cv::flip(original, mirror, 1);

	std::cout << "--mirror" << std::endl;
	std::cout << mirror << std::endl;

	std::cout << "--mirror90" << std::endl;
	rotate_90n(mirror, temp, 90);
	std::cout << temp << std::endl;

	std::cout << "--mirror180" << std::endl;
	rotate_90n(mirror, temp, 180);
	std::cout << temp << std::endl;

	std::cout << "--mirror270" << std::endl;
	rotate_90n(mirror, temp, 270);
	std::cout << temp << std::endl;


	if (ENABLE_DEBUG_IMAGE){
		cv::Mat test_image = cv::imread("resource\\transform_generator\\test.png", cv::IMREAD_UNCHANGED);
		cv::Mat test_mirror;
		cv::Mat test_temp;

		if (test_image.empty()){
			std::cout << "test image load failed" << std::endl;
			return;
		}
		cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
		cv::imshow("test", test_image);
		cv::waitKey(0);

		//90
		rotate_90n(test_image, test_temp, 90);
		cv::imshow("test", test_temp);
		cv::waitKey(0);

		//180
		rotate_90n(test_image, test_temp, 180);
		cv::imshow("test", test_temp);
		cv::waitKey(0);

		//270
		rotate_90n(test_image, test_temp, 270);
		cv::imshow("test", test_temp);
		cv::waitKey(0);

		//mirror
		cv::flip(test_image, test_mirror, 1);
		cv::imshow("test", test_mirror);
		cv::waitKey(0);

		//mirror90
		rotate_90n(test_mirror, test_temp, 90);
		cv::imshow("test", test_temp);
		cv::waitKey(0);

		//mirror180
		rotate_90n(test_mirror, test_temp, 180);
		cv::imshow("test", test_temp);
		cv::waitKey(0);

		//mirror270
		rotate_90n(test_mirror, test_temp, 270);
		cv::imshow("test", test_temp);
		cv::waitKey(0);

		cv::destroyAllWindows();
	}

}