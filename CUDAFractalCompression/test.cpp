#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <bitset>

#include "test.hpp"
#include "config.hpp"
#include "images.hpp"
#include "csv_util.hpp"
#include "ifs_transform_data.hpp"

#include "fif_util.hpp"

/*
	bit_pack, unpackが正常に動作するかのテスト
*/
void bit_packing_test(Images& images) {

	std::vector<uint64_t> ifs_packed_data;

	ifs_header header;
	std::vector<ifs_transformer*> ifs_data;
	
		header.image_height =512;
		header.image_width = 512;
		//ifs_data = fc.compress(entry.second);
	
		import_csv("lena-512x512.png-mse0.csv", ifs_data);
		
		//std::cout << "before : " << (uint64_t)ifs_data[10]->affine << std::endl;

		print_ifs_all(ifs_data, 10);

		std::cout << "bit pack testing..." << std::endl;
		ifs_pack(&header, ifs_data, ifs_packed_data, false);

		std::cout << "packed ifs length : " << ifs_packed_data.size() << std::endl;
		std::cout << "packed ifs size : " << ifs_packed_data.size() * sizeof(uint64_t) << " bytes:" << std::endl;

		/*
			数列のままファイルを保存してみる

		*/
		std::ofstream ofs("lena-512x512.png-mse0-integer-array.txt", std::ios::out | std::ios::binary | std::ios::trunc);

		if (!ofs) {
			std::cerr << "csv出力に失敗" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		for (uint64_t data : ifs_packed_data) {
			ofs.write((char*)&data, sizeof(uint64_t));
		}
		ofs << std::endl;

		std::cout << "bit unpack testing..." << std::endl;
		ifs_unpack(&header, ifs_packed_data, ifs_data, false);

		//std::cout << "after" << (uint64_t)ifs_data[10]->affine << std::endl;

		print_ifs_all(ifs_data, 10);

		
	//後始末
	for (ifs_transformer* c : ifs_data) {
		delete c;
	}
}
