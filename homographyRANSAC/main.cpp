#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include "data_reader.h"
#include "ransac.h"

 

int main(int argc, const char** argv)
{
    srand(time(NULL));

    std::string path = argv[1];
    std::string path_save = argv[2];
    std::vector<std::string> file_paths = ListDirectory(path);
    std::string& path_panoramic_images = file_paths.at(0);

    std::string window_name = "image";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    size_t ratio = 2;
    cv::Mat
        img1,
        img2;

    const size_t 
        k_sample_size = 4,
        n_iterations = 1000;
    const double 
        threshold = 3,
        confidence = 0.95;

    std::cout << path_panoramic_images << std::endl;
    std::vector<std::string> path_images = ListDirectory(path_panoramic_images);

    int res_img = 1;

    for (auto& it : path_images)
    {
        std::vector<std::string> images = ListDirectory(it);

        for (size_t i = 1; i < images.size(); ++i)
        {
            img1 = cv::imread(images.at(i - 1));
            img2 = cv::imread(images.at(i));
            cv::resize(img1, img1, cv::Size(img1.cols / ratio, img1.rows / ratio));
            cv::resize(img2, img2, cv::Size(img2.cols / ratio, img2.rows / ratio));

            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            std::vector<cv::DMatch> good_matches;
            MatchFeatures(img1, img2, keypoints1, keypoints2, good_matches);
            cv::Mat img_matches;
            cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);
            //-- Show detected matches
            imshow(window_name, img_matches);
            cv::moveWindow(window_name, 0, 0);
            cv::waitKey(0);
            cv::destroyWindow(window_name);

            std::vector<cv::Point2d> points_img1, points_img2;
            points_img1.reserve(good_matches.size());
            points_img2.reserve(good_matches.size());

            for (size_t i = 0; i < good_matches.size(); ++i)
            {
                cv::Point2d p1 = keypoints1[good_matches[i].queryIdx].pt;
                cv::Point2d p2 = keypoints2[good_matches[i].trainIdx].pt;
                points_img1.emplace_back(p1);
                points_img2.emplace_back(p2);
            }

            cv::Mat best_matrix_H;
            std::vector<size_t> best_inliers;

            FindHomographyRANSAC(points_img1, points_img2, k_sample_size, best_matrix_H,
                best_inliers, threshold, n_iterations, confidence);

            std::cout << "matrix H: " << std::endl << best_matrix_H << std::endl;

            // Use the Homography Matrix to warp the images
            cv::Mat result;
            cv::warpPerspective(img2, result, best_matrix_H.inv(),
                cv::Size(img1.cols + img2.cols, img1.rows));
            cv::Mat half(result, cv::Rect(0, 0, img2.cols, img2.rows));
            img1.copyTo(half);
            imshow(window_name, result);
            cv::moveWindow(window_name, 0, 0);
            cv::waitKey(0);

            std::string ps = path_save + std::to_string(res_img) + std::string(".jpg");
            cv::imwrite(ps, result);
            res_img += 1;
        }
    }
    return 0;
}
