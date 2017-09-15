#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <opencv2\ml.hpp>
#include <string>
#include "Log.h"
#include "ImagePreProcess.h"
#include <opencv2\ml.hpp>

double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0) {
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}




/* findSquares: returns sequence of squares detected on the image
*/
void findSquares(const cv::Mat& src, std::vector<std::vector<cv::Point> >& squares) {
	cv::Mat src_gray;
	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

	// Blur helps to decrease the amount of detected edges
	cv::Mat filtered;
	cv::blur(src_gray, filtered, cv::Size(3, 3));
	cv::imwrite("out_blur.jpg", filtered);

	// Detect edges
	cv::Mat edges;
	int thresh = 64;
	cv::Canny(filtered, edges, thresh, thresh * 2, 3);
	cv::imwrite("out_edges.jpg", edges);

	// Dilate helps to connect nearby line segments
	cv::Mat dilated_edges;
	cv::dilate(edges, dilated_edges, cv::Mat(), cv::Point(-1, -1), 2, 1, 1); // default 3x3 kernel
	cv::imwrite("out_dilated.jpg", dilated_edges);

	// Find contours and store them in a list
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(dilated_edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	// Test contours and assemble squares out of them
	std::vector<cv::Point> approx;
	for (size_t i = 0; i < contours.size(); i++) {
		// approximate contour with accuracy proportional to the contour perimeter
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);

		// Note: absolute value of an area is used because
		// area may be positive or negative - in accordance with the
		// contour orientation
		if (approx.size() == 4 && std::fabs(contourArea(cv::Mat(approx))) > 1000 &&
			cv::isContourConvex(cv::Mat(approx))) {
			double maxCosine = 0;
			for (int j = 2; j < 5; j++) {
				double cosine = std::fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
				maxCosine = MAX(maxCosine, cosine);
			}

			if (maxCosine < 0.3)
				squares.push_back(approx);
		}
	}
}

/* findLargestSquare: find the largest square within a set of squares
*/
void findLargestSquare(const std::vector<std::vector<cv::Point> >& squares,
	std::vector<cv::Point>& biggest_square) {
	if (!squares.size()) {
		std::cout << "findLargestSquare !!! No squares detect, nothing to do." << std::endl;
		return;
	}

	int max_width = 0;
	int max_height = 0;
	int max_square_idx = 0;

	cv::Point start, end;
	for (size_t i = 0; i < squares.size(); i++) {
		// Convert a set of 4 unordered Points into a meaningful cv::Rect structure.
		cv::Rect rectangle = cv::boundingRect(cv::Mat(squares[i]));

		//std::cout << "find_largest_square: #" << i << " rectangle x:" << rectangle.x << " y:" << rectangle.y << " " << rectangle.width << "x" << rectangle.height << endl;

		// Store the index position of the biggest square found
		if ((rectangle.width >= max_width) && (rectangle.height >= max_height)) {
			max_width = rectangle.width;
			max_height = rectangle.height;
			max_square_idx = i;
			start.x = rectangle.x;
			start.y = rectangle.y;
			end.x = rectangle.width;
			end.y = rectangle.height;
		}
	}
	biggest_square.push_back(start);
	biggest_square.push_back(end);
}

int main() {
	std::string picPath = R"(C:\Users\Fan\Desktop\resize.jpg)";
	cv::Mat src = cv::imread(picPath);
	std::cout << src.size().width << std::endl;
	cv::imshow("source", src);

	std::vector<std::vector<cv::Point> > squares;
	findSquares(src, squares);

	std::vector<cv::Point> largest_square;
	findLargestSquare(squares, largest_square);
	std::cout << largest_square.size() << std::endl;
	if (largest_square.size()) {

		//printf("%d,%d,%d,%d\n", largest_square[0].x, largest_square[0].y, largest_square[1].x, largest_square[1].y);
		auto imageRoi = src(cv::Rect(largest_square[0].x, largest_square[0].y, largest_square[1].x, largest_square[1].y));
		cv::imshow("Corners", imageRoi);
		cv::imwrite("result.jpg", imageRoi);
		cv::Mat imageRplane = getRplane(imageRoi);

		std::vector <cv::RotatedRect> rects;
		postDetect(imageRplane, rects);

		cv::Mat outputMat;
		normalPosArea(imageRplane, rects[0], outputMat);

		std::vector<cv::Mat> charMat;
		charSegment(outputMat, charMat);

		getAnnXml();


		cv::Ptr< cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
		annTrain(ann, 10, 24);

		std::vector<int> charResult;
		classify(ann, charMat, charResult);

		for (int i : charResult) {
			util::log(i);
		}
	}


	cv::waitKey(0);
	return 0;
}