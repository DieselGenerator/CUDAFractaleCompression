#pragma once

#include <iostream>
#include <vector>
#include <string>

/*
	�o�͉摜(�t���N�^���摜)�ɕK�v�ȏ���ێ����Ă���
*/
struct ifs_header {
	uint32_t image_width;
	uint32_t image_height;
};

/*
	�o�͉摜(�t���N�^���摜)�̔����n�֐�
	�����o���̍ۂ͂��������ďo�͂���
*/
struct ifs_transformer {
	//�Ή�����u���b�N��id���L�����Ă���
	//���ۂɏ����o���ۂ͕��ׂ�����id�����ʂł���̂ŁC�����Wid�͂���Ȃ�
	uint32_t dblock_x;
	uint32_t dblock_y;

	uint32_t rblock_x;
	uint32_t rblock_y;
	
	//0 = 0.0625
	//1 = 0.125
	//2 = 0.1875
	//15= 1
	//�R���g���X�g�X�P�[�����O
	uint8_t scaling;
	
	//�P�x�V�t�g
	uint8_t shift;
	
	//0b001 = 90�x��]
	//0b010 = 180�x��]
	//0b100 = �����ϊ�
	//ex) 0b111 �̏ꍇ��(90 + 180)�x��] + �����ϊ��ƂȂ�
	uint8_t affine;

	//�u���b�N�̑傫��
	uint8_t blocksize;

	//mse
	double error;
};

void print_ifs_header(ifs_header*);
void print_ifs_data(ifs_transformer*);
void print_ifs_all(std::vector<ifs_transformer*>, uint32_t);
