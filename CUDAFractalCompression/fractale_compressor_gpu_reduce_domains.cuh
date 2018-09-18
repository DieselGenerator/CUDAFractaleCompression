/*
	GPU�̃h���C���������炵������
*/

#pragma once

#include <opencv2/core.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ifs_transform_data.hpp"


/*
	�u���b�N���̃X���b�h�������̒l�ɂȂ�悤�ɓw�͂���
	1024�ȉ��ł���K�v������
*/
#define THREADBLOCK_MAX 1024

struct compress_data_part_reduce_domains_gpu {
	//�Ή�����u���b�N��id���L�����Ă���
	//���ۂɏ����o���ۂ͕��ׂ�����id�����ʂł���̂ŁC�����Wid�͂���Ȃ�
	uint32_t dblock_id;
	uint32_t rblock_id;
	double scale;
	uint8_t shift;
	//true�ŋ���
	bool mirror;
	//0 = 0', 1 = 90', 2 = 180', 3 = 270'
	uint8_t rotate;
};

std::vector<ifs_transformer*> launch_reduce_domains_compress_kernel(cv::Mat img, uint32_t blocksize, uint32_t range_count);
