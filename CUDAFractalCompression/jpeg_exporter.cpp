#include <iostream>

#include <boost/filesystem.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "jpeg_exporter.hpp"
#include "comparators.hpp"

void export_jpeg(std::string filename,  const cv::Mat original) {
	const boost::filesystem::path dir("out\\jpeg");
	if (!(boost::filesystem::exists(dir))) {
		boost::filesystem::create_directory(dir);
	}

	for (int quality = 0; quality <= 100; quality++) {

		std::vector<uchar> buff;//buffer for coding
		std::vector<int> param = std::vector<int>(2);
		param[0] = CV_IMWRITE_JPEG_QUALITY;
		param[1] = quality;//default(95) 0-100

		cv::imencode(".jpg", original, buff, param);
		cv::Mat jpegimage = cv::imdecode(cv::Mat(buff), CV_LOAD_IMAGE_UNCHANGED);

		double psnr = calcPSNR(original, jpegimage);
		std::cout << "Quality : " << quality << " PSNR : " << psnr << std::endl;
		cv::imwrite("out\\jpeg\\" + filename + "-quality-" + std::to_string(quality) + ".jpg", jpegimage);

	}

}
