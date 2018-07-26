#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "affine_transformer.hpp"

/*
input:
	const cv::Mat& domain		:	�􉽕ϊ���K�p����h���C���u���b�N
	const uint8_t affine_pattern:	�A�t�B���ϊ��̃p�^�[�����w�肷��
output:
	cv::Mat& derived_domain		:	�􉽕ϊ���K�p�����h���C���u���b�N��ۑ�����
*/
void affine_transform(
	const cv::Mat& domain,
	const uint8_t affine_pattern,
	cv::Mat& derived_domain) {

	//�h�������h���C���u���b�N��ۑ�����
	derived_domain = domain.clone();

	//�����ϊ�
	if (0b100 & affine_pattern) {
		cv::flip(domain, derived_domain, 1);
	}
	
	//90�x��]
	if ((0b011 & affine_pattern) == 0b001) {
		cv::transpose(derived_domain, derived_domain);
		cv::flip(derived_domain, derived_domain, 1);
		//std::cout << "90" << std::endl;
	}
	//180�x��]
	else if ((0b011 & affine_pattern) == 0b010) {
		cv::flip(derived_domain, derived_domain, -1);
		//std::cout << "180" << std::endl;
	}
	//270�x��]
	else if ((0b011 & affine_pattern) == 0b011) {
		cv::transpose(derived_domain, derived_domain);
		cv::flip(derived_domain, derived_domain, 0);
		//std::cout << "270" << std::endl;
	};
	/*debug�p*/
	//std::cout << "pattern" << (uint32_t)affine_pattern << std::endl;
	//cv::namedWindow("Image", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	//cv::imshow("Image", derived_domain);
	//cv::waitKey(0);
}

/*
input:
	const cv::Mat& input		:	����Mat
	const uint32_t depth		:	���{�k�����邩1��1/2�{, 2��1/4�{�k��
output:
	cv::Mat& half		:	����Mat���k�����ďo�͂���
*/
void resize_half(
	const cv::Mat& input,
	cv::Mat& half,
	const uint32_t depth) {

	cv::Size resize_cvsize = cv::Size(input.cols >> depth, input.rows >> depth);
	cv::resize(input, half, resize_cvsize, cv::INTER_NEAREST);
}
