#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include <boost/filesystem.hpp>

#include "ifs_transform_data.hpp"

#include "csv_util.hpp"

void import_csv(std::string filename, std::vector<ifs_transformer*>& compressed_data){

	const char delimiter = ',';
	std::fstream filestream(filename);
	if (!filestream.is_open()){
		std::cerr << "csv入力に失敗" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	//行数
	uint64_t counter = 0;

	// ファイルを読み込む
	while (!filestream.eof())
	{
		counter++;
		// １行読み込む
		std::string buffer;
		filestream >> buffer;
		
		//4行目まではヘッダー扱いとする
		if(counter < 6){
			std::cout << buffer << std::endl;
			continue;
		}
		
		// ファイルから読み込んだ１行の文字列を区切り文字で分けてリストに追加する
		std::vector<std::string> record;              // １行分の文字列のリスト
		std::istringstream streambuffer(buffer); // 文字列ストリーム
		std::string token;                       // １セル分の文字列

		while (std::getline(streambuffer, token, delimiter))
		{
			// １セル分の文字列をリストに追加する
			record.push_back(token);
		}

		//std::cout << record.size() << std::endl;
		if(record.size() != 9){
			return;
		}

		ifs_transformer* ifs = new ifs_transformer();
		ifs->dblock_x = (uint32_t)std::stoi(record[0]);
		ifs->dblock_y = (uint32_t)std::stoi(record[1]);
		ifs->rblock_x = (uint32_t)std::stoi(record[2]);
		ifs->rblock_y = (uint32_t)std::stoi(record[3]);
		ifs->blocksize = (uint8_t)std::stoi(record[4]);
		ifs->affine = (uint8_t)std::stoi(record[5]);
		ifs->scaling = (uint8_t)std::stoi(record[6]);
		ifs->shift = (uint8_t)std::stoi(record[7]);
		ifs->error = std::stod(record[8]);

		// １行分の文字列を出力引数のリストに追加する
		compressed_data.push_back(ifs);
	}

}

void export_csv(std::string name, std::vector<ifs_transformer*>& compressed_data, long long encode_time, long long decode_time = -1, double psnr = -1) {
	const boost::filesystem::path dir("out");
	const std::string prefix("out\\");
	const std::string suffix(".csv");
	const std::string csv_name(prefix + name + suffix);

	if (!(boost::filesystem::exists(dir))) {
		boost::filesystem::create_directory(dir);
	}

	std::ofstream ofs(csv_name);

	if (!ofs) {
		std::cerr << "csv出力に失敗" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	ofs << "encode time" << "," << encode_time << std::endl;
	ofs << "decode_time" << "," << decode_time << std::endl;
	ofs << "PSNR" << "," << psnr << std::endl;

	ofs << "dx" << "," << "dy" << ","
		<< "rx" << "," << "ry" << ","
		<< "blocksize" << ","
		<< "affine" << ","
		<< "scaling" << ","
		<< "shifting" << ","
		<< "error" << std::endl;

	for (ifs_transformer* ifs : compressed_data) {
		ofs << ifs->dblock_x << "," << ifs->dblock_y << ","
			<< ifs->rblock_x << "," << ifs->rblock_y << ","
			<< (uint32_t)ifs->blocksize << ","
			<< (uint32_t)ifs->affine << ","
			<< (uint32_t)ifs->scaling << ","
			<< (uint32_t)ifs->shift << ","
			<< ifs->error << std::endl;
	}

	uint32_t count = 0;

	//ウィンドウの最大表示数

}
