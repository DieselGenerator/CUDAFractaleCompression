//�R���X�^���g�������̓O���[�o���ϐ��ɂł��Ȃ�
//�Ƃ������Ƃŕʃt�@�C������A�N�Z�X�ł��Ȃ��ׁCdeprecated
/*
	�􉽕ϊ��ׂ̈̍s��𐶐�����
	�܂�GPU�ň����ׂɃR���X�^���g�������ւ̓]�����s��
*/

#pragma once

#include <inttypes.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
	�R���X�^���g�������Ŋ􉽕ϊ���index��ێ�����
	affine_transform_sizen[0] = 90�� ��]
	affine_transform_sizen[0] = 180����]
	affine_transform_sizen[0] = 270����]
	affine_transform_sizen[0] = ���� 0�� ��] 
	affine_transform_sizen[0] = ���� 90����]
	affine_transform_sizen[0] = ���� 180����]
	affine_transform_sizen[0] = ���� 270����]
*/
//extern __device__ uint8_t dc_affine_transform_size4_1d[7*16];
//
//extern __device__ uint8_t dc_affine_transform_size4[7][16];
//extern __device__ uint8_t dc_affine_transform_size8[7][64];
//extern __device__ uint8_t dc_affine_transform_size16[7][256];
//
//void init_affine_transformer(int size);
