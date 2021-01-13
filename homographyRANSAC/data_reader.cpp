#include "data_reader.h"


std::vector<std::string> ListDirectory(std::string& path)
{
    std::vector<std::string> file_paths;

    for (const auto& entry : fs::directory_iterator(path))
        file_paths.push_back(entry.path().string());

    return file_paths;
}


void ReadData(
    std::string& path,
    std::vector<cv::Point2d>& points_img1,
    std::vector<cv::Point2d>& points_img2
)
{
    std::string line;
    std::ifstream PointFile(path);

    std::cout << "Reading feature matched 2D point coordinates from path: " 
        << std::endl << path << std::endl;

    while (std::getline(PointFile, line))
    {
        double x1, y1, x2, y2;
        std::string s_x1, s_y1, s_x2, s_y2;
        std::stringstream ss(line);

        ss >> s_x1 >> s_y1 >> s_x2 >> s_y2;
        x1 = atof(s_x1.c_str());
        y1 = atof(s_y1.c_str());
        x2 = atof(s_x2.c_str());
        y2 = atof(s_y2.c_str());

        cv::Point2d p1{ x1, y1};
        cv::Point2d p2{ x2, y2 };

        points_img1.emplace_back(p1);
        points_img2.emplace_back(p2);
    }
    PointFile.close();

    std::cout << "Finished reading from file." << std::endl;
}