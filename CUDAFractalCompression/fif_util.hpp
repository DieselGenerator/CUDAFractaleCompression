/*
	オリジナルの圧縮形式である.fif形式への圧縮を行う．
	圧縮の過程でbitpacking/huffman codingを行う
*/

#pragma once

#include <iostream>
#include <vector>

#include "ifs_transform_data.hpp"

//テストコード用
void ifs_pack(ifs_header*, std::vector<ifs_transformer*>&, std::vector<uint64_t>&, bool);
void ifs_unpack(ifs_header*, std::vector<uint64_t>&, std::vector<ifs_transformer*>&, bool);

void fif_compress(std::string, std::vector<ifs_transformer*>&);
void fif_decompress(std::vector<ifs_transformer*>&);