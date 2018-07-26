#pragma once

#include <iostream>
#include <vector>
#include <string>

/*
	出力画像(フラクタル画像)に必要な情報を保持しておく
*/
struct ifs_header {
	uint32_t image_width;
	uint32_t image_height;
};

/*
	出力画像(フラクタル画像)の反復系関数
	書き出しの際はこれを削って出力する
*/
struct ifs_transformer {
	//対応するブロックのidを記憶しておく
	//実際に書き出す際は並べた順でidを識別できるので，レンジidはいらない
	uint32_t dblock_x;
	uint32_t dblock_y;

	uint32_t rblock_x;
	uint32_t rblock_y;
	
	//0 = 0.0625
	//1 = 0.125
	//2 = 0.1875
	//15= 1
	//コントラストスケーリング
	uint8_t scaling;
	
	//輝度シフト
	uint8_t shift;
	
	//0b001 = 90度回転
	//0b010 = 180度回転
	//0b100 = 鏡像変換
	//ex) 0b111 の場合は(90 + 180)度回転 + 鏡像変換となる
	uint8_t affine;

	//ブロックの大きさ
	uint8_t blocksize;

	//mse
	double error;
};

void print_ifs_header(ifs_header*);
void print_ifs_data(ifs_transformer*);
void print_ifs_all(std::vector<ifs_transformer*>, uint32_t);
