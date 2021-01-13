#pragma once
#include <vector>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include <iostream>


void MatchFeatures(
	cv::Mat& img1,
	cv::Mat& img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& good_matches
);

size_t GetIterationNumber(
	const double& inlier_ratio,
	const double& confidence,
	const size_t& sample_size
);

void SelectMinimalSample
(
	size_t& n_points,
	std::vector<size_t>& sample,
	const size_t& k_sample_size
);

std::vector<cv::Point2d> NormalizePoints(
	std::vector<cv::Point2d>& points, 
	std::vector<size_t>& indices,
	cv::Mat& translation,
	cv::Mat& scale
);

cv::Mat GetMatrixA(
	std::vector<cv::Point2d>& normalized_points_img1,
	std::vector<cv::Point2d>& normalized_points_img2,
	const size_t& k_sample_size
);

cv::Mat GetProjectionMatrix(cv::Mat& matrix_A);

cv::Mat CalculateHomographyMatrix(
	std::vector<cv::Point2d>& points_img1,
	std::vector<cv::Point2d>& points_img2,
	std::vector<size_t>& indices
);

void CalculateInliers(
	std::vector<cv::Point2d>& points_img1,
	std::vector<cv::Point2d>& points_img2,
	cv::Mat& matrix_H,
	std::vector<size_t>& current_inliers
);

void FindHomographyRANSAC(
	std::vector<cv::Point2d>& points_img1,
	std::vector<cv::Point2d>& points_img2,
	const size_t& k_sample_size,
	cv::Mat& best_matrix_H,
	std::vector<size_t> best_inliers_idx,
	const double& threshold,
	const size_t& n_iterations,
	const double& confidence
);

void checkHomographyCorrectness(
	std::vector<cv::Point2d>& normalized_points_img1,
	std::vector<cv::Point2d>& normalized_points_img2,
	cv::Mat& matrix_H
);

