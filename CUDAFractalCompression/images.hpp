/*
	�摜�f�[�^
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
	//�摜�ۑ��t�H���_
	std::string prefix = "resource\\";
	//�O���[�X�P�[���̏o�͈ʒu
	std::string grey_prefix = "grey\\";
public:
	//�摜�f�[�^��ێ�����
	//�z�X�g��
	std::map<std::string, cv::Mat> h_images;
	//�f�o�C�X��
	std::map<std::string, cv::cuda::GpuMat> d_images;

	//�O���[�X�P�[���̉摜���ێ����Ă���
	//�z�X�g��
	std::map<std::string, cv::Mat> h_grey_images;
	//�f�o�C�X��
	std::map<std::string, cv::cuda::GpuMat> d_grey_images;

	/*---------------------------------------------------------*/

	//�R���X�g���N�^
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
