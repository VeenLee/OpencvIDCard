#include "ImagePreProcess.h"
#include <opencv2\ml\ml.hpp>
#include <opencv\cv.h>
#include <opencv2\imgproc.hpp>
#include<opencv2\opencv.hpp>
#include<opencv2\core.hpp>
#include <fstream>
#include "Log.h"



using namespace cv;

Mat getRplane(const Mat & in) {
	std::vector<Mat> splitRGB(in.channels());
	cv::split(in, splitRGB);
	return splitRGB[2];
}

void ostuBeresenThreshold(const Mat&in, Mat &out) {

	double ostu_T = threshold(in, out, 0, 255, CV_THRESH_OTSU);
	double min;
	double max;
	minMaxIdx(in, &min, &max);
	const double CI = 0.32;
	double beta = CI* (max - min + 1) / 128;
	double beta_lowT = (1 - beta)*ostu_T;
	double beta_highT = (1 + beta) * ostu_T;

	Mat doubleMatIn;
	in.copyTo(doubleMatIn);
	int rows = doubleMatIn.rows;
	int cols = doubleMatIn.cols;
	double Tbn;

	for (int i = 0; i < rows; i++) {
		uchar *p = doubleMatIn.ptr<uchar>(i);
		uchar *outPtr = out.ptr<uchar>(i);
		for (int j = 0; j < cols; j++) {
			if (i < 2 || i > rows - 3 || j < 2 || j > rows - 3) {
				if (p[j] <= beta_lowT) {
					outPtr[j] = 0;
				} else {
					outPtr[j] = 255;
				}
			} else {
				Tbn = sum(doubleMatIn(Rect(i - 2, j - 2, 5, 5)))[0] / 25;
				if (p[j] < beta_lowT || (p[j] < Tbn && (beta_lowT <= p[j] && p[j] >= beta_highT))) {
					outPtr[j] = 0;
				}
				if (p[j] > beta_highT || p[j] >= Tbn && (beta_lowT <= p[j] && p[j] >= beta_highT)) {
					outPtr[j] = 255;
				}
			}
		}
	}
}

bool isEligible(const RotatedRect &candidate) {
	float error = 0.2;
	const float aspect = 4.5 / 0.3;
	int min = 10 * aspect * 10;
	int max = 50 * aspect * 50;
	float rmin = aspect - aspect * error;
	float rmax = aspect + aspect * error;

	int area = candidate.size.height *candidate.size.width;
	float r = (float)candidate.size.width / (float)candidate.size.height;
	if (r < 1) {
		r = 1 / r;
	}

	if ((area < min || area > max) || (r < rmin || r > rmax)) {
		return false;
	} else {
		return true;
	}
}


void postDetect(const cv::Mat & in, std::vector<cv::RotatedRect>& rects) {
	Mat threshold_R;
	ostuBeresenThreshold(in, threshold_R);

	//imshow("ostu", threshold_R);

	Mat imgInv(in.size(), in.type(), Scalar(255));
	Mat threshold_Inv = imgInv - threshold_R;


	Mat element = getStructuringElement(MORPH_RECT, Size(15, 3));
	morphologyEx(threshold_Inv, threshold_Inv, CV_MOP_CLOSE, element);

	std::vector<std::vector<Point>> contours;

	imshow("threshold_Inv", threshold_Inv);
	findContours(threshold_Inv, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	auto itc = contours.begin();

	util::log("contours_size", (int)contours.size());

	int i = 0;
	while (itc != contours.end()) {
		RotatedRect mr = minAreaRect(Mat(*itc));
		if (!isEligible(mr)) {
			itc = contours.erase(itc);
		} else {
			rects.push_back(mr);
			++itc;
		}
		++i;
	}
}

void normalPosArea(const cv::Mat & inputImg, cv::RotatedRect & rects_optimal, cv::Mat & output_area) {
	float r, angle;
	angle = rects_optimal.angle;
	r = (float)rects_optimal.size.width / (float)rects_optimal.size.height;
	if (r < 1) {
		angle = 90 + angle;
	}
	Mat rotMat = getRotationMatrix2D(rects_optimal.center, angle, 1);
	Mat img_rotated;
	warpAffine(inputImg, img_rotated, rotMat, inputImg.size(), CV_INTER_CUBIC);

	Size rect_size = rects_optimal.size;

	if (r < 1) {
		std::swap(rect_size.width, rect_size.height);
	}

	Mat img_crop;
	getRectSubPix(img_rotated, rect_size, rects_optimal.center, img_crop);

	Mat resultResized;
	resultResized.create(20, 300, CV_8UC1);

	resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);

	resultResized.copyTo(output_area);
}

void charSegment(const cv::Mat & inputImg, std::vector<cv::Mat>& dest_mat) {
	Mat img_threshold;
	util::log("charSegment");
	//util::log(inputImg.rows);
	//util::log(inputImg.cols);

	imshow("charSegment", inputImg);
	imwrite("result\\idcode.jpg", inputImg);

	Mat whiteImage(inputImg.size(), inputImg.type(), Scalar(255));
	Mat in_Inv = whiteImage - inputImg;

	threshold(in_Inv, img_threshold, 0, 255, CV_THRESH_OTSU);

	int x_char[19] = { 0 };
	short counter = 1;
	short num = 0;
	bool *flag = new bool[img_threshold.cols];

	for (int j = 0; j < img_threshold.cols; j++) {
		flag[j] = true;
		for (int i = 0; i < img_threshold.rows; i++) {
			if (img_threshold.at<uchar>(i, j) != 0) {
				flag[j] = false;
				break;
			}
		}
	}

	for (int i = 0; i < img_threshold.cols - 2; i++) {
		if (flag[i] == true) {
			x_char[counter] += i;
			num++;
			if (!flag[i + 1] && !flag[i + 2]) {
				x_char[counter] = x_char[counter] / num;
				num = 0;
				counter++;
			}
		}
	}
	x_char[18] = img_threshold.cols;

	//util::log()

	for (int i = 0; i < 18; i++) {
		Rect  rect(x_char[i], 0, x_char[i + 1] - x_char[i], img_threshold.rows);
		std::cout << i << ":" << "x:" << rect.x << "y:" << rect.y << "width:" << rect.width << "height:" << rect.height << std::endl;
		dest_mat.push_back(Mat(in_Inv, rect));
	}
	delete[] flag;
}

float sumMatValue(const Mat &image) {
	float sumValue = 0;
	int r = image.rows;
	int c = image.cols;

	if (image.isContinuous()) {
		c = r *c;
		r = 1;
	}

	for (int i = 0; i < r; i++) {
		const uchar *linePtr = image.ptr<uchar>(i);
		for (int j = 0; j < c; j++) {
			sumValue += linePtr[j];
		}
	}
	return sumValue;
}

Mat projectHistogram(const Mat &img, int t) {
	Mat lowData;
	resize(img, lowData, Size(8, 16));

	int sz = (t) ? lowData.rows : lowData.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j < sz; j++) {
		Mat data = (t) ? lowData.row(j) : lowData.col(j);
		mhist.at<float>(j) = countNonZero(data);
	}

	double min, max;
	minMaxLoc(mhist, &min, &max);

	if (max > 0) {
		mhist.convertTo(mhist, -1, 1.0f / max, 0);
	}
	return mhist;
}

void calcGradientFeat(const Mat &imgSrc, Mat &out) {
	std::vector <float> feat;
	Mat image;

	resize(imgSrc, image, Size(8, 16));

	float mask[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };

	Mat y_mask = Mat(3, 3, CV_32F, mask) / 8;
	Mat x_mask = y_mask.t();
	Mat sobelX, sobelY;

	filter2D(image, sobelX, CV_32F, x_mask);
	filter2D(image, sobelY, CV_32F, y_mask);

	sobelX = abs(sobelX);
	sobelY = abs(sobelY);

	float totalValueX = sumMatValue(sobelX);
	float totalValueY = sumMatValue(sobelY);

	for (int i = 0; i < image.rows; i += 4) {
		for (int j = 0; j < image.cols; j += 4) {
			Mat subImageX = sobelX(Rect(j, i, 4, 4));
			feat.push_back(sumMatValue(subImageX) / totalValueX);
			Mat subImageY = sobelY(Rect(j, i, 4, 4));
			feat.push_back(sumMatValue(subImageY) / totalValueY);
		}
	}

	Mat imageGray;
	resize(imgSrc, imageGray, Size(4, 8));
	Mat p = imageGray.reshape(1, 1);
	p.convertTo(p, CV_32FC1);
	for (int i = 0; i < p.cols; i++) {
		feat.push_back(p.at<float>(i));
	}

	Mat vhist = projectHistogram(imgSrc, 1);
	Mat hhist = projectHistogram(imgSrc, 0);
	for (int i = 0; i < vhist.cols; i++) {
		feat.push_back(vhist.at<float>(i));
	}
	for (int i = 0; i < hhist.cols; i++) {
		feat.push_back(hhist.at<float>(i));
	}
	out = Mat::zeros(1, feat.size(), CV_32F);
	for (int i = 0; i < feat.size(); i++) {
		out.at<float>(i) = feat[i];
	}
}

void getAnnXml() {
	std::ifstream fin("ann.xml");
	if (fin) {
		return;
	}
	FileStorage fs("ann.xml", FileStorage::WRITE);

	if (!fs.isOpened()) {
	}

	Mat trainData;
	Mat classes = Mat::zeros(1, 550, CV_8UC1);
	char path[60];
	Mat img_read;
	for (int i = 0; i < 10; i++) {
		for (int j = 1; j < 51; j++) {
			sprintf_s(path, "Number_char/%d/%d (%d).png", i, i, j);
			img_read = imread(path, 0);
			Mat dest_feature;
			calcGradientFeat(img_read, dest_feature);
			trainData.push_back(dest_feature);

			classes.at<uchar>(i * 50 + j - 1) = i;
		}
	}
	fs << "TrainingData" << trainData;
	fs << "classes" << classes;
	fs.release();
}

void annTrain(cv::Ptr<cv::ml::ANN_MLP> ann, int numCharacters, int nlayers) {
	Mat trainData, classes;
	FileStorage fs;
	fs.open("ann.xml", FileStorage::READ);

	fs["TrainingData"] >> trainData;
	fs["classes"] >> classes;

	Mat layerSizes(1, 3, CV_32SC1);
	layerSizes.at<int>(0) = trainData.cols;
	layerSizes.at<int>(1) = nlayers;
	layerSizes.at<int>(2) = numCharacters;
	//ann->create(layerSizes, ml::ANN_MLP::SIGMOID_SYM, 1, 1);
	ann->setLayerSizes(layerSizes);
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1, 1);
	TermCriteria termCirteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 5000, 0.01);
	ann->setTermCriteria(termCirteria);

	Mat trainClasses;
	trainClasses.create(trainData.rows, numCharacters, CV_32FC1);
	for (int i = 0; i < trainData.rows; i++) {
		for (int k = 0; k < trainClasses.cols; k++) {
			if (k == (int)classes.at<uchar>(i)) {
				trainClasses.at<float>(i, k) = i;
			} else {
				trainClasses.at<float>(i, k) = 0;
			}
		}
	}
	ann->train(trainData, ml::ROW_SAMPLE, trainClasses);
}

void classify(cv::Ptr<cv::ml::ANN_MLP> ann, std::vector<cv::Mat>& charMat, std::vector<int>& charResult) {
	charResult.resize(charMat.size());
	for (int i = 0; i < charMat.size(); i++) {
		Mat outPut(1, 10, CV_32FC1);

		std::ostringstream name;
		name <<"result\\"<< i << ".jpg";
		imwrite(name.str(), charMat[i]);
		Mat charFature;
		calcGradientFeat(charMat[i], charFature);
		ann->predict(charFature, outPut);

		Point maxLoc;
		double maxVal;
		minMaxLoc(outPut, 0, &maxVal, 0, &maxLoc);
		charResult[i] = maxLoc.x;
	}
}
