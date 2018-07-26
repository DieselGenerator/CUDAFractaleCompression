/*
	!deprecated
	圧縮データの取扱いを行う
	実際に圧縮する前の全体のデータ及び圧縮した結果を書き出す部分を司る
*/

#pragma once

#include <iostream>

/*
	保存用構造体
	書き出しの際はこれを削って出力する
*/
struct compressed_data {
	//対応するブロックのidを記憶しておく
	//実際に書き出す際は並べた順でidを識別できるので，レンジidはいらない
	uint32_t dblock_id;
	uint32_t rblock_id;
	
	//コンスタントスケーリング
	double scale;
	
	//輝度シフト
	uint8_t shift;
	
	//trueで鏡像
	bool mirror;
	
	//0 = 0', 1 = 90', 2 = 180', 3 = 270'
	uint8_t rotate;

	//mse
	double error;
};
