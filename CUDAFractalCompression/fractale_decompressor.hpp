/*
	フラクタル展開を行うクラス。
	何故クラスにした？意味不明。
*/

#pragma once

#include <iostream>
#include <vector>

#include "ifs_transform_data.hpp"

class FractaleDecompressor {
public:
	cv::Mat decompress(ifs_header&, std::vector<ifs_transformer*>&);
};