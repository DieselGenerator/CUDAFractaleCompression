#undef NDEBUG//assert�p

#include <cassert>

#include "config.hpp"

/*
	�f�o�b�O�Ɋւ���ݒ�
*/
//�d����GPU�̏���N�����ɘa���邽�߂̑[�u�����{���邩
const bool config::enable_gpu_warmup = true;
//GPU�̏����o�͂��邩
const bool config::enable_print_device_info = false;
//���Z�e�X�g�i�����j���s����
const bool config::enable_arithmetic_speedtest = false;
//���O�ɗ��p����ׂ̊􉽕ϊ��e�[�u���𐶐����Ă�����
const bool config::enable_transform_generator = false;

/*
	CPU��OpenCV��p�������k�̐ݒ�
*/
//CPU�ł�OpenCV�𗘗p�������k���s����
const bool config::enable_cpu_opencv_compress = false;
//CPU�ł�OpenCV�𗘗p�����L�����s�����i�������̈��k���s����K�v������j
const bool config::enable_cpu_opencv_decompress = true;
//CPU�ł�OpenCV�𗘗p�������k�摜��W�J�������̂��o�͂��邩�i�������̐L�����s����K�v������j
const bool config::enable_cpu_opencv_output = true;;
//���k���s���ۂ�臒l
const double config::mse_threshold = 0;
//�l���ؕ������s����
const bool config::enable_quartree_compress = true;
//range,domain�̑傫���̐ݒ�
const uint8_t config::range_size_min = 4;
const uint8_t config::range_size_max = 16;
const uint8_t config::domain_size_min = 8;
const uint8_t config::domain_size_max = 32;

/*
	GPU��OpenCV��p�������k�̐ݒ�(������)
*/
//GPU�ł�OpenCV�𗘗p�������k���s����
const bool config::enable_gpu_opencv_compress = false;

/*
	GPU�ŃI���W�i���J�[�l����p�������k�̐ݒ�
*/
//GPU�̃I���W�i���J�[�l���𗘗p�������k���s����
const bool config::enable_gpu_compress = false;
//GPU�̃I���W�i���J�[�l���𗘗p�����L�����s�����i�������̈��k���s����K�v������j
const bool config::enable_gpu_decompress = true;
//GPU�̃I���W�i���J�[�l���𗘗p�������k�摜��W�J�������̂��o�͂��邩�i�������̐L�����s����K�v������j
const bool config::enable_gpu_output = true;;

/*
	GPU�Ńh���C�������J�[�l����p�������k�̐ݒ�
*/
//GPU�̃h���C�������J�[�l���𗘗p�������k���s����
const bool config::enable_gpu_reduce_domains_compress = false;
//GPU�̃h���C�������J�[�l���𗘗p�����L�����s�����i�������̈��k���s����K�v������j
const bool config::enable_gpu_reduce_domains_decompress = true;
//GPU�̃h���C�������J�[�l���𗘗p�������k�摜��W�J�������̂��o�͂��邩�i�������̐L�����s����K�v������j
const bool config::enable_gpu_reduce_domains_output = true;

/*
	GPU�Ń����W�����J�[�l����p�������k�̐ݒ�
*/
//GPU�̃����W�����J�[�l���𗘗p�������k���s����
const bool config::enable_gpu_reduce_ranges_compress = true;
//GPU�̃����W�����J�[�l���𗘗p�����L�����s�����i�������̈��k���s����K�v������j
const bool config::enable_gpu_reduce_ranges_decompress = true;
//GPU�̃����W�����J�[�l���𗘗p�������k�摜��W�J�������̂��o�͂��邩�i�������̐L�����s����K�v������j
const bool config::enable_gpu_reduce_ranges_output = true;

/*
	CPU�ł�libJpeg��p�������k�̐ݒ�
*/
const bool config::enable_jpeg_compress = false;
const bool config::enable_jpeg_output = false;
const bool config::enable_jpeg_repeat = false;
const bool config::enable_jpeg_output_statistics = false;

const bool config::show_decompress_process = false;
const bool config::output_decompress_process_image = false;
const bool config::output_decompress_image = true;

const bool config::show_demo = false;

/*
	CUDA�Ɋւ���ݒ�
*/
const uint32_t thread_pre_block = 1024;

static_assert(config::range_size_min * 2 == config::domain_size_min, "range");

void config_assert() {
	assert(config::range_size_min * 2 == config::domain_size_min);
	assert(config::range_size_max * 2 == config::domain_size_max);
	assert(thread_pre_block <= 1024);
}
