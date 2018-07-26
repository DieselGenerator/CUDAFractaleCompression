/*
	�I���W�i���̈��k�`���ł���.fif�`���ւ̈��k���s���D
	���k�̉ߒ���bitpacking/huffman coding���s��
*/

#pragma once

#include <iostream>
#include <vector>

#include "ifs_transform_data.hpp"

//�e�X�g�R�[�h�p
void ifs_pack(ifs_header*, std::vector<ifs_transformer*>&, std::vector<uint64_t>&, bool);
void ifs_unpack(ifs_header*, std::vector<uint64_t>&, std::vector<ifs_transformer*>&, bool);

void fif_compress(std::string, std::vector<ifs_transformer*>&);
void fif_decompress(std::vector<ifs_transformer*>&);