#pragma once
#include <opencv\cv.h>
#include <opencv2\ml.hpp>

cv::Mat getRplane(const cv::Mat &in);

void postDetect(const cv::Mat &in, std::vector<cv::RotatedRect> &rects);

void normalPosArea(const cv::Mat &inputImg, cv::RotatedRect &rects_optimal, cv::Mat &output_area);

void charSegment(const cv::Mat &inputImg, std::vector<cv::Mat> &dest_mat);

void getAnnXml();

void annTrain(cv::Ptr<cv::ml::ANN_MLP> ann, int numCharacters, int nlayers);

void classify(cv::Ptr<cv::ml::ANN_MLP> ann, std::vector<cv::Mat> &charMat, std::vector<int> &charResult);