/*
	!deprecated
	���k�f�[�^�̎戵�����s��
	���ۂɈ��k����O�̑S�̂̃f�[�^�y�ш��k�������ʂ������o���������i��
*/

#pragma once

#include <iostream>

/*
	�ۑ��p�\����
	�����o���̍ۂ͂��������ďo�͂���
*/
struct compressed_data {
	//�Ή�����u���b�N��id���L�����Ă���
	//���ۂɏ����o���ۂ͕��ׂ�����id�����ʂł���̂ŁC�����Wid�͂���Ȃ�
	uint32_t dblock_id;
	uint32_t rblock_id;
	
	//�R���X�^���g�X�P�[�����O
	double scale;
	
	//�P�x�V�t�g
	uint8_t shift;
	
	//true�ŋ���
	bool mirror;
	
	//0 = 0', 1 = 90', 2 = 180', 3 = 270'
	uint8_t rotate;

	//mse
	double error;
};
