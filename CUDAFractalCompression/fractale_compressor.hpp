/*
	�t���N�^�����k���s���N���X�B
	���̃N���X�ɂ����H�Ӗ��s���B�Ȃ����B
*/

#pragma once

#include <iostream>
#include <vector>
#include <map>

#include <boost/dynamic_bitset.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "compress_data.hpp"
#include "ifs_transform_data.hpp"

//�h���C���T�C�Y(DOMAIN_SIZE * DOMAIN_SIZE�̉摜�����)
#define DOMAIN_SIZE 8
//�����W�T�C�Y(RANGE_SIZE * RANGE_SIZE�̉摜�����)
#define RANGE_SIZE 4

struct bufferMSE{
	//CUDA�p
	//gpu��Ƀ����������蓖�Ă�̂͂ƂĂ��R�X�g������
	//�����ōŏ��ɍ�������̂��g���܂킷�ׂ�
	cv::cuda::GpuMat d_gs, d_t1, d_t2;
	cv::cuda::GpuMat d_buf;
};

class FractaleCompressor {
public:
	FractaleCompressor();

	std::vector<ifs_transformer*> compress(cv::Mat);

	std::vector<compressed_data> compress(cv::cuda::GpuMat);

	double calcMSE(const cv::Mat&, const cv::Mat&);

private:
	template<typename MAT>
	uint32_t countDomain(MAT h_mat);

	uint8_t contrast_scaling(const cv::Mat&, const cv::Mat&);
	uint8_t brightness_shift(const cv::Mat&, const cv::Mat&);

	void ifs_init(uint32_t, uint32_t, uint8_t, ifs_transformer*);
	//void affine_transform(const cv::Mat&, const uint8_t, cv::Mat&);
	void ifs_transform(cv::Mat, cv::Mat, ifs_transformer*);

	//double calcMSE(const cv::Mat&, const cv::Mat&);
	double calcMSEoptimized(cv::cuda::GpuMat, cv::cuda::GpuMat, bufferMSE&);
};
