#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "affine_transformer.hpp"
#include "config.hpp"
#include "comparators.hpp"

#include "fractale_decompressor.hpp"

cv::Mat FractaleDecompressor::decompress(ifs_header& header, std::vector<ifs_transformer*>& compressed_data) {

	//初期化画像の生成
	cv::Mat decoded_image = cv::Mat::zeros(header.image_width, header.image_height, CV_8U);
	for (int j = 0; j < decoded_image.rows; j++) {
		for (int i = 0; i < decoded_image.cols; i++) {
			if (i >= (header.image_width >> 1)) {
				decoded_image.at<uint8_t>(j, i) = 255;
			}
		}
	}

	if (config::show_demo) {
		cv::namedWindow("decoded", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		cv::namedWindow("codebook", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	}

	cv::Mat resized;

	cv::Mat before_step_decoded_image;
	double before_step_psnr = 0;
	double after_step_psnr = 0;
	uint32_t step_counter = 0;

	//cv::Mat resized4, resized8, resized16;
	//展開
	do {
		step_counter++;
		before_step_psnr = after_step_psnr;
		before_step_decoded_image = decoded_image.clone();

		resize_half(decoded_image, resized, 1);

		//resize_half(decoded_image, resized4, 1);
		//resize_half(decoded_image, resized8, 1);
		//resize_half(decoded_image, resized16, 1);

		for (ifs_transformer* ifs : compressed_data) {
			//TODO ROIをうまく使いたい
			//TODO rangeサイズ

			if (config::show_demo) {
				cv::Mat demo_range;
				//(decoded_image.clone()).convertTo(demo_range, CV_8UC3);
				cv::cvtColor((decoded_image.clone()), demo_range, CV_GRAY2RGB);

				//std::cout << demo_range.type() << std::endl;
				cv::rectangle(demo_range, cv::Rect(ifs->rblock_x, ifs->rblock_y, ifs->blocksize, ifs->blocksize), cv::Scalar(0, 0, 255), 1);

				cv::Mat demo_domain;
				//(resized.clone()).convertTo(demo_domain, CV_8UC3);
				cv::cvtColor((resized.clone()), demo_domain, CV_GRAY2RGB);
				cv::rectangle(demo_domain, cv::Rect(ifs->dblock_x, ifs->dblock_y, ifs->blocksize, ifs->blocksize), cv::Scalar(0, 0, 255), 1);

				cv::putText(demo_domain, "scaling : " + std::to_string(ifs->scaling), cv::Point(0, 20), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 255), 2, CV_AA);
				cv::putText(demo_domain, "shift : " + std::to_string(ifs->shift), cv::Point(0, 40), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 255), 2, CV_AA);
				cv::putText(demo_domain, "affine : " + std::to_string(ifs->affine), cv::Point(0, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 255), 2, CV_AA);
				cv::putText(demo_domain, "blocksize : " + std::to_string(ifs->blocksize), cv::Point(0, 80), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 255), 2, CV_AA);

				cv::imshow("decoded", demo_range);
				cv::imshow("codebook", demo_domain);

				cv::waitKey(1);
			}
			
			cv::Rect domain_roi(ifs->dblock_x, ifs->dblock_y, ifs->blocksize, ifs->blocksize);
			cv::Mat derived_domain;
			//std::cout << ifs->dblock_x << ":" << ifs->dblock_y << std::endl;
			derived_domain = resized(domain_roi);

			//if (ifs->blocksize == 4){
			//	derived_domain = resized4(domain_roi);
			//}
			//else if(ifs->blocksize == 8){
			//	derived_domain = resized8(domain_roi);
			//}
			//else if(ifs->blocksize == 16){
			//	derived_domain = resized16(domain_roi);
			//}

			//cv::Mat derived_domain = resized(domain_roi);

			affine_transform(resized(domain_roi), ifs->affine, derived_domain);

			//if (ifs->blocksize == 4) {
			//	affine_transform(resized4(domain_roi), ifs->affine, derived_domain);
			//}
			//else if (ifs->blocksize == 8) {
			//	affine_transform(resized8(domain_roi), ifs->affine, derived_domain);
			//}
			//else if (ifs->blocksize == 16) {
			//	affine_transform(resized16(domain_roi), ifs->affine, derived_domain);
			//}
			
			derived_domain = derived_domain * ((ifs->scaling + 1) * 0.0625) + ifs->shift;
	
			cv::Rect range_roi(ifs->rblock_x, ifs->rblock_y, ifs->blocksize, ifs->blocksize);
	
			derived_domain.copyTo(decoded_image(range_roi));

		}
		const std::string winname = "Decode:out\\fractale " + std::to_string(step_counter) + ".png";

		if (config::show_decompress_process) {
			cv::namedWindow(winname);
			cv::imshow(winname, decoded_image);
			cv::waitKey();
			cv::destroyWindow(winname);
		}

		if (config::output_decompress_process_image){
			cv::imwrite("out\\farctale " + std::to_string(step_counter) + ".png", decoded_image);
		}

		if (config::show_demo) {
			cv::waitKey();
		}

		after_step_psnr = calcPSNR(decoded_image, before_step_decoded_image);

		//std::cout << step_counter << std::endl;

	} while (before_step_psnr != after_step_psnr && step_counter < 200);

	std::cout << "decode step count : " << step_counter << std::endl;

	if (config::show_demo) {
		cv::destroyWindow("decoded");
		cv::destroyWindow("codebook");
	}

	return decoded_image;
}
