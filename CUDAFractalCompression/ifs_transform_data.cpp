/*
	ifsデータの表示・その他utilを担う
*/

#include <iostream>
#include <string>

#include "ifs_transform_data.hpp"

void print_ifs_header(ifs_header* ifs_header) {

}

void print_ifs_data(ifs_transformer*) {

}

void print_ifs_all(std::vector<ifs_transformer*>ifs_data, uint32_t limit) {
	std::cout << "----- print_ifs_all ------" << std::endl;

	uint64_t counter = 0;

	for (ifs_transformer* ifs : ifs_data) {
		if (counter > limit) {
			break;
		}
		std::cout << "count : " << counter << std::endl;
		std::cout << "dx : "<< ifs->dblock_x << ", dy :" << ifs->dblock_y 
				  << ", affine : " << (uint32_t)ifs->affine 
				  << ", scaling : " << (uint32_t)ifs->scaling
				  << ", shifting :" << (uint32_t)ifs->shift << std::endl;
		counter++;
	}
}
