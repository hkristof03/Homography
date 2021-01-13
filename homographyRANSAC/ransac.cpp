#include "ransac.h"


void MatchFeatures(
	cv::Mat& img1,
	cv::Mat& img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& good_matches
)
{
	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
	const float ratio_thresh = 0.7f;
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
	cv::Mat descriptors1, descriptors2;

	detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
	std::cout << "Detected " << keypoints1.size() << " keypoints on the first image!"
		<< std::endl << "Detected " << keypoints2.size() 
		<< " keypoints on the second image!" << std::endl;
	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
	// Since SURF is a floating-point descriptor NORM_L2 is used
	cv::Ptr<cv::DescriptorMatcher> matcher =
		cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<cv::DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	//-- Filter matches using the Lowe's ratio test
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	std::cout << "Detected " << good_matches.size() << " corresponding points!" << std::endl;
}

size_t GetIterationNumber(
	const double& inlier_ratio,
	const double& confidence,
	const size_t& sample_size
)
{
	double a = log(1.0 - confidence);
	double b = log(1.0 - std::pow(inlier_ratio, sample_size));

	if (abs(b) < std::numeric_limits<double>::epsilon())
		return std::numeric_limits<double>::epsilon();

	size_t it_num = static_cast<size_t>(a / b);

	return it_num;
}

void SelectMinimalSample
(
	size_t& n_points,
	std::vector<size_t>& sample,
	const size_t& k_sample_size
)
{
	sample.clear();
	// 1. Select a minimal sample -> 4 random points for homography
	while (sample.size() != k_sample_size)
	{
		// Generate a random index between [0, n_points]
		size_t random_idx =
			size_t((n_points - 1.0) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

		if (std::find(sample.begin(), sample.end(), random_idx) == sample.end())
			sample.emplace_back(random_idx);
	}
}

std::vector<cv::Point2d> NormalizePoints(
	std::vector<cv::Point2d>& points, 
	std::vector<size_t>& indices,
	cv::Mat& translation,
	cv::Mat& scale
)
{
	std::vector<cv::Point2d> normalized_points;
	normalized_points.reserve(indices.size());
	// Calculate the mass point
	cv::Point2d mass_point(0, 0);
	translation = cv::Mat::eye(3, 3, CV_64F);
	scale = cv::Mat::eye(3, 3, CV_64F);

	for (const auto& idx : indices)
	{
		mass_point += points.at(idx);
		normalized_points.emplace_back(points.at(idx));
	}
	mass_point *= (1.0 / indices.size());
	// Move the points to have the origin in their mass point 
	// and calculate the average distance from the origin
	double avg_dist = 0.0;

	for (auto& p : normalized_points)
	{
		p -= mass_point;
		avg_dist += cv::norm(p);
	}
	avg_dist /= normalized_points.size();
	// Set up the translation matrix to unnormalize matrix_H later
	translation.at<double>(0, 2) = -mass_point.x;
	translation.at<double>(1, 2) = -mass_point.y;
	// Make the average distance for each point from the mass to be sqrt(2)
	const double ratio = sqrt(2) / avg_dist;
	for (auto& p : normalized_points)
		p *= ratio;
	// Set up the scale matrix to unnormalize matrix_H later
	scale.at<double>(0, 0) = ratio;
	scale.at<double>(1, 1) = ratio;

	return normalized_points;
}

cv::Mat GetMatrixA(
	std::vector<cv::Point2d>& normalized_points_img1,
	std::vector<cv::Point2d>& normalized_points_img2,
	const size_t& k_sample_size
)
{
	cv::Mat matrix_A;

	for (size_t i = 0; i < k_sample_size; ++i)
	{
		cv::Mat vector = cv::Mat::zeros(2, 9, CV_64F);
		cv::Point2d& p1 = normalized_points_img1[i];
		cv::Point2d& p2 = normalized_points_img2[i];

		vector.at<double>(0, 0) = p1.x;
		vector.at<double>(0, 1) = p1.y;
		vector.at<double>(0, 2) = 1.0;
		vector.at<double>(0, 6) = -p1.x * p2.x; 
		vector.at<double>(0, 7) = -p1.y * p2.x;
		vector.at<double>(0, 8) = -p2.x;

		vector.at<double>(1, 3) = p1.x;
		vector.at<double>(1, 4) = p1.y;
		vector.at<double>(1, 5) = 1.0;
		vector.at<double>(1, 6) = -p1.x * p2.y;
		vector.at<double>(1, 7) = -p1.y * p2.y;
		vector.at<double>(1, 8) = -p2.y;

		matrix_A.push_back(vector);
	}
	return matrix_A;
}

cv::Mat GetProjectionMatrix(cv::Mat& matrix_A)
{
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(matrix_A.t() * matrix_A, eigenvalues, eigenvectors);
	cv::Mat matrix_H = eigenvectors.row(eigenvectors.rows -1);
	// Normalize matrix H
	matrix_H = matrix_H * (1.0 / matrix_H.at<double>(eigenvectors.rows -1));

	return matrix_H;
}

cv::Mat CalculateHomographyMatrix(
	std::vector<cv::Point2d>& points_img1,
	std::vector<cv::Point2d>& points_img2,
	std::vector<size_t>& indices
)
{
	const size_t sample_size = indices.size();
	cv::Mat translation1, translation2, scale1, scale2;
	std::vector<cv::Point2d> normalized_points_img1 =
		NormalizePoints(points_img1, indices, translation1, scale1);
	std::vector<cv::Point2d> normalized_points_img2 =
		NormalizePoints(points_img2, indices, translation2, scale2);
	// Get matrix A formed from the selected n-n points
	cv::Mat matrix_A = GetMatrixA(normalized_points_img1,
		normalized_points_img2, sample_size);
	// Calculate the projection matrix H by getting the eigenvector
	// corresponding to the lowest eigenvalue of A^T*T and reshape it to 3x3
	cv::Mat matrix_H = GetProjectionMatrix(matrix_A);
	matrix_H = matrix_H.reshape(1, 3);

	cv::Mat T1 = scale1 * translation1;
	cv::Mat T2 = scale2 * translation2;
	//Unnormalize the Homography matrices
	matrix_H = T2.inv() * matrix_H * T1;
	// Check homography's correctness ---------------------------------------
	//checkHomographyCorrectness(normalized_points_img1, normalized_points_img2,
	//	matrix_H);
	// -----------------------------------------------------------------------
	// Check the unnormalized homography's correctness -----------------------
	//std::vector<cv::Point2d> selected_points1, selected_points2;
	//selected_points1.reserve(sample_indices.size());
	//selected_points2.reserve(sample_indices.size());
	//
	//for (const auto& idx : sample_indices)
	//{
	//	selected_points1.emplace_back(points_img1.at(idx));
	//	selected_points2.emplace_back(points_img2.at(idx));
	//}
	//checkHomographyCorrectness(selected_points1, selected_points2, matrix_H);
	// -----------------------------------------------------------------------
	return matrix_H;
}

void CalculateInliers(
	std::vector<cv::Point2d>& points_img1,
	std::vector<cv::Point2d>& points_img2,
	cv::Mat& matrix_H,
	const double& threshold,
	std::vector<size_t>& current_inliers
)
{
	current_inliers.clear();
	cv::Mat homogeneous_points1, homogeneous_points2;

	cv::convertPointsToHomogeneous(points_img1, homogeneous_points1);
	cv::convertPointsToHomogeneous(points_img2, homogeneous_points2);
	cv::Mat matrix_H_inv = matrix_H.inv();

	for (size_t idx = 0; idx < homogeneous_points1.rows; ++idx)
	{
		cv::Mat v1 = homogeneous_points1.row(idx).reshape(1, 3);
		cv::Mat v2 = homogeneous_points2.row(idx).reshape(1, 3);

		cv::Mat v1_prime = matrix_H * v1;
		cv::Mat v2_prime = matrix_H_inv * v2;

		double v1_prime_x = v1_prime.at<double>(0, 0) / v1_prime.at<double>(2, 0);
		double v1_prime_y = v1_prime.at<double>(1, 0) / v1_prime.at<double>(2, 0);
		double v2_prime_x = v2_prime.at<double>(0, 0) / v2_prime.at<double>(2, 0);
		double v2_prime_y = v2_prime.at<double>(1, 0) / v2_prime.at<double>(2, 0);

		cv::Point2d v1_prime_cartesian = { v1_prime_x, v1_prime_y };
		cv::Point2d v2_prime_cartesian = { v2_prime_x, v2_prime_y };

		double distance = cv::norm(points_img2.at(idx) - v1_prime_cartesian)
			+ cv::norm(points_img1.at(idx) - v2_prime_cartesian);

		if (distance < threshold)
		{
			current_inliers.emplace_back(idx);
		}
	}
}

void FindHomographyRANSAC(
	std::vector<cv::Point2d>& points_img1,
	std::vector<cv::Point2d>& points_img2,
	const size_t& k_sample_size,
	cv::Mat& best_matrix_H,
	std::vector<size_t> best_inliers,
	const double& threshold,
	const size_t& max_iterations,
	const double& confidence
)
{
	// set random seed
	srand(time(NULL));
	size_t n_points = points_img1.size();
	// The current number of iterations
	size_t iteration_number = 0;
	size_t n_iterations = max_iterations;
	// The indices of the inliers of the current best model
	std::vector<size_t> current_inliers;
	current_inliers.reserve(points_img1.size());
	// The current sample indices
	std::vector<size_t> sample_indices;
	sample_indices.reserve(k_sample_size);

	std::cout << "Searching for Homography with RANSAC!" << std::endl
		<< "Number of found point correspondences: " << points_img1.size() 
		<< std::endl << "Threshold is: " << threshold << std::endl
		<< "Performing " << max_iterations << " iterations." << std::endl;

	while (iteration_number++ < n_iterations)
	{
		if (iteration_number % 10 == 0)
			std::cout << "Current iteration: " << iteration_number << std::endl;
		SelectMinimalSample(n_points, sample_indices, k_sample_size);
		// collinearity check here....
		// Translation and Scale matrices
		cv::Mat matrix_H = CalculateHomographyMatrix(points_img1, 
			points_img2, sample_indices);
		cv::Mat matrix_H_inv = matrix_H.inv();
		// Count the number of inliers
		CalculateInliers(points_img1, points_img2, matrix_H, 
			threshold, current_inliers);

		if (current_inliers.size() > best_inliers.size())
		{
			std::cout << "Iteration number: " << iteration_number << std::endl 
				<< "Current best inliers size: " << current_inliers.size() 
				<< std::endl;

			best_inliers.swap(current_inliers);
			best_matrix_H = matrix_H.clone();
			current_inliers.clear();
			current_inliers.resize(0);
		}
		// Update the maximum iteration number
		double inlier_ratio = static_cast<double>(best_inliers.size()) /
			static_cast<double>(points_img1.size());
		n_iterations = GetIterationNumber(
			inlier_ratio,
			confidence,
			k_sample_size
		);
	}
	best_matrix_H = CalculateHomographyMatrix(points_img1, points_img2, 
		best_inliers);
}

void checkHomographyCorrectness(
	std::vector<cv::Point2d>& points_img1,
	std::vector<cv::Point2d>& points_img2,
	cv::Mat& matrix_H
)
{
	cv::Mat homogeneous_points1, homogeneous_points2;
	std::vector<cv::Point3d> hom_proj_points1, hom_proj_points2;
	std::vector<cv::Point2d> inh_proj_points1, inh_proj_points2;

	inh_proj_points1.reserve(points_img1.size());
	inh_proj_points2.reserve(points_img2.size());

	cv::Mat matrix_H_inv = matrix_H.inv();

	cv::convertPointsToHomogeneous(points_img1, homogeneous_points1);
	cv::convertPointsToHomogeneous(points_img2, homogeneous_points2);
	
	for (size_t idx = 0; idx < homogeneous_points1.rows; ++idx)
	{
		cv::Mat v1 = homogeneous_points1.row(idx).reshape(1, 3);
		cv::Mat v2 = homogeneous_points2.row(idx).reshape(1, 3);

		cv::Mat v1_prime = matrix_H_inv * v2;
		cv::Mat v2_prime = matrix_H * v1;

		hom_proj_points1.emplace_back(cv::Point3d(v1_prime.at<double>(0, 0),
			v1_prime.at<double>(1, 0), v1_prime.at<double>(2, 0)));
		hom_proj_points2.emplace_back(cv::Point3d(v2_prime.at<double>(0, 0),
			v2_prime.at<double>(1, 0), v2_prime.at<double>(2, 0)));
		
	}
	cv::convertPointsFromHomogeneous(hom_proj_points1, inh_proj_points1);
	cv::convertPointsFromHomogeneous(hom_proj_points2, inh_proj_points2);

	for (size_t idx = 0; idx < points_img1.size(); ++idx)
	{
		std::cout << "v1: " << points_img1.at(idx) << std::endl << "v1_prime: "
			<< inh_proj_points1.at(idx) << std::endl << "v2: " << points_img2.at(idx) 
			<< std::endl << "v2_prime" << inh_proj_points2.at(idx) << std::endl;
	}
}
