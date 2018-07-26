#include <iostream>
#include <cmath>
#include <bitset>
#include <vector>

#include "ifs_transform_data.hpp"
#include "fif_util.hpp"

/*
	ifs����bit��ɕϊ�(uint64_t��array)�ɋl�ߍ���
	packet�̍\����
	useQuadTree == false�̎�
	|x�ʒu( log2(width/4) bit)|y�ʒu( log2(height/4) bit)|scaling(4bit)|shifting(8bit)|affine(3bit)|

*/
void ifs_pack(ifs_header* ifs_header, std::vector<ifs_transformer*>& ifs_data, std::vector<uint64_t>& ifs_packed_data, bool useQuadTree) {

	if (useQuadTree) {
		std::cerr << "Not implemented fif_util - ifs_pack" << std::endl;
	}

	//�e�p�����[�^�Ŏg��bit�̐�
	const uint32_t x_bits = std::log2(ifs_header->image_width) - 2;
	const uint32_t y_bits = std::log2(ifs_header->image_height) - 2;
	const uint32_t scaling_bits = 4;
	const uint32_t shifting_bits = 8;
	const uint32_t affine_bits = 3;
	const uint32_t total_packet_bits = x_bits + y_bits + scaling_bits + shifting_bits + affine_bits;

	std::cout << "xbits : " << x_bits << std::endl;
	std::cout << "ybits : " << y_bits << std::endl;
	std::cout << "scaling bits : " << scaling_bits << std::endl;
	std::cout << "shifting bits : " << shifting_bits << std::endl;
	std::cout << "affine bits : " << affine_bits << std::endl;
	std::cout << "total packet bits : " << total_packet_bits << std::endl;

	//���݂̃p�P�b�g�͉�bit�l�܂��Ă��邩
	uint32_t used_bits = 0;
	ifs_packed_data.clear();

	//ifs_packed_data�ɒǉ�����packet���i�[���Ă���
	for (ifs_transformer* ifs : ifs_data) {

		//�Ƃ肠����uint64_t�̃p�P�b�g�𐶐�����
		uint64_t packet = 0;
		//4�{�̏������
		packet += (ifs->dblock_x >> 2);
		packet <<= y_bits;
		packet += (ifs->dblock_y >> 2);
		packet <<= scaling_bits;
		packet += ifs->scaling;
		packet <<= shifting_bits;
		packet += ifs->shift;
		packet <<= affine_bits;
		packet += ifs->affine;

		//std::cout << "testing : " << std::bitset<64>(packet) <<std::endl;

		//���݂�uint64_t���Ɏ��܂�Ȃ��̂ł����vector��ǉ�����
		if (64 < used_bits + total_packet_bits) {
			uint64_t& last_ifs = ifs_packed_data.back();
			//�c�艽bit���邩
			uint32_t front_bits = 64 - used_bits;
			//���̃p�P�b�g��bit�̐�
			uint32_t back_bits = total_packet_bits - front_bits;
			uint64_t front_packet = packet >> (total_packet_bits - front_bits);
			uint64_t back_packet = packet << (64 - back_bits);
			last_ifs += front_packet;
			ifs_packed_data.push_back(back_packet);
			used_bits = back_bits;
		}
		//���݂̃x�N�^�Ɏ��܂�ꍇ
		else {
			//����̂�
			if (ifs_packed_data.empty()) {
				//���l��
				ifs_packed_data.push_back(packet << (64 - total_packet_bits));
				used_bits = total_packet_bits;
			}
			//packet���V�t�g���đ}������
			else {
				uint64_t& lastpacket = ifs_packed_data.back();
				lastpacket += (packet <<= (64 - used_bits - total_packet_bits));
				used_bits += total_packet_bits;
			}
		}

	}
}

void ifs_unpack(ifs_header* ifs_header, std::vector<uint64_t>& ifs_packed_data, std::vector<ifs_transformer*>& ifs_data, bool useQuadTree) {

	if (useQuadTree) {
		std::cerr << "Not implemented fif_util - ifs_unpack" << std::endl;
	}

	//�e�p�����[�^�Ŏg��bit�̐�
	const uint32_t x_bits = std::log2(ifs_header->image_width) - 2;
	const uint32_t y_bits = std::log2(ifs_header->image_height) - 2;
	const uint32_t scaling_bits = 4;
	const uint32_t shifting_bits = 8;
	const uint32_t affine_bits = 3;
	const uint32_t total_packet_bits = x_bits + y_bits + scaling_bits + shifting_bits + affine_bits;

	//���݂̓ǂݐ؂���bit��
	uint32_t read_bits = 0;
	//�J��Ԃ��ۂɕK�v�Ȏc��bit��
	uint32_t remain_bits = 0;
	//�J��z���p�P�b�g�ۑ��p
	uint64_t temp_packet = 0;

	ifs_data.clear();
	for (uint64_t packet : ifs_packed_data) {

		//getchar();

		//�O����̌J�z������ꍇ
		if (temp_packet) {
			temp_packet += (packet >> (64 - remain_bits));
			
			ifs_transformer* ifs = new ifs_transformer();
			ifs_data.push_back(ifs);
			ifs->affine = temp_packet & 0b111;
			ifs->shift = (temp_packet & 0b11111111000) >> affine_bits;
			ifs->scaling = (temp_packet & 0b111100000000000) >> (affine_bits + shifting_bits);
			ifs->dblock_y = (temp_packet & ((uint64_t)(std::pow(2, y_bits) - 1) << (scaling_bits + shifting_bits + affine_bits))) >> (affine_bits + shifting_bits + scaling_bits);
			ifs->dblock_x = (temp_packet & ((uint64_t)(std::pow(2, x_bits) - 1) << (scaling_bits + shifting_bits + affine_bits + y_bits))) >> (affine_bits + shifting_bits + scaling_bits + y_bits);

			temp_packet = 0;
			read_bits = remain_bits;
			remain_bits = 0;
		}

		//�ǂݐ؂��p�P�b�g���܂�����ꍇ
		while ((64 - read_bits) > total_packet_bits){

			ifs_transformer* ifs = new ifs_transformer();
			ifs_data.push_back(ifs);
			uint32_t read_start = read_bits == 0 ? UINT32_MAX : std::pow(2, 64 - read_bits);

			//packet�̕s�v�ȉE��������
			uint64_t fpacket = packet >> ((64 - read_bits) - total_packet_bits);
			//packet�̕s�v�ȍ���������
			ifs->affine = fpacket & 0b111;
			ifs->shift = (fpacket & 0b11111111000) >> affine_bits;
			ifs->scaling = (fpacket & 0b111100000000000) >> (affine_bits + shifting_bits);
			ifs->dblock_y = (fpacket & ((uint64_t)(std::pow(2, y_bits) - 1) << (scaling_bits + shifting_bits + affine_bits))) >> (affine_bits + shifting_bits + scaling_bits);
			ifs->dblock_x = (fpacket & ((uint64_t)(std::pow(2, x_bits) - 1) << (scaling_bits + shifting_bits + affine_bits + y_bits))) >> (affine_bits + shifting_bits + scaling_bits + y_bits);
		
			read_bits += total_packet_bits;
		}

		//�J�z�ɔ�����
		remain_bits = total_packet_bits - (64 - read_bits);
		//std::cout << "remain : " << remain_bits << std::endl;

		temp_packet = ((packet & ((uint64_t)std::pow(2, total_packet_bits - remain_bits) - 1)) << remain_bits);

		read_bits = 0;
		
	}
}

/*
input:
	std::string filename					:	���k��̃t�@�C����
	std::vector<ifs_transformer*>& ifs_data :	���k����ifs�f�[�^

��̓I�ɂ�bitpacking/huffman coding���s���C�t�@�C���Ƃ��ĕۑ�����
*/
void fif_compress(std::string filename, std::vector<ifs_transformer*>& ifs_data) {

}

void fif_decompress(std::vector<ifs_transformer*>& ifs_data, std::string filename) {

}
