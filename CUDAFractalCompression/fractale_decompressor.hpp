/*
	�t���N�^���W�J���s���N���X�B
	���̃N���X�ɂ����H�Ӗ��s���B
*/

#pragma once

#include <iostream>
#include <vector>

#include "ifs_transform_data.hpp"

class FractaleDecompressor {
public:
	cv::Mat decompress(ifs_header&, std::vector<ifs_transformer*>&);
};