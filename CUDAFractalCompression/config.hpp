/*
	�e��ݒ��ێ�����
	�l��ύX���������ōăR���p�C�����������Ȃ��̂Ŏ��̂Ə����l���\�[�X�ɂ�����
*/

#pragma once

#include <cinttypes>

struct config {
	/*
		�f�o�b�O�Ɋւ���ݒ�
	*/
	static const bool enable_gpu_warmup;
	static const bool enable_print_device_info;
	static const bool enable_arithmetic_speedtest;
	static const bool enable_transform_generator;

	static const bool enable_cpu_opencv_compress;
	static const bool enable_cpu_opencv_decompress;

	static const bool enable_gpu_opencv_compress;

	static const bool enable_gpu_compress;
	static const bool enable_gpu_reduce_domains_compress;
	static const bool enable_gpu_reduce_ranges_compress;

	static const bool enable_jpeg_compress;
	static const bool enable_jpeg_output;
	static const bool enable_jpeg_repeat;
	static const bool enable_jpeg_output_statistics;

	static const bool show_decompress_process;
	static const bool output_decompress_process_image;
	static const bool output_decompress_image;

	static const bool show_demo;
	/*
		�t���N�^�����k�Ɋւ���ݒ�
	*/
	//const�ɂ����̂͊ԈႢ�������C�ϒl�ɂ���
	static const double config::mse_threshold;
	//�l���򕪊����s��Ȃ��ꍇ��range_size_max�̒l�ŕ��������
	static const bool enable_quartree_compress;

	//���̂�������
	static const uint8_t range_size;
	static const uint8_t domain_size;

	static const uint8_t range_size_min;
	static const uint8_t range_size_max;

	static const uint8_t domain_size_min;
	static const uint8_t domain_size_max;

	/*
		CUDA�Ɋւ���ݒ�
	*/
	static const uint32_t thread_pre_block;
};

void config_assert();
