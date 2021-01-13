#pragma once
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <iostream>


namespace fs = std::filesystem;

std::vector<std::string> ListDirectory(std::string& path);

void ReadData(
	std::string& path,
	std::vector<cv::Point2d>& points_img1,
	std::vector<cv::Point2d>& points_img2
);