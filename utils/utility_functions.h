/*
  Varun Anusheel, Lokesh Saipureddi
  Spring 2024
  CS 5330 Computer Vision

  Project 2: Content-Based Image Retrieval.
*/
#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using std::vector;
using std::string;
using namespace cv;

vector<double> computeFeatureVector(Mat& image, const string& matchingType);
void visualizeHist(Mat& hist);
bool fileExists(const string& filename);
double computeIntersection(vector<double>& hist1, vector<double>& hist2);
double computeDistance(const vector<double>& featureVector1, const vector<double>& featureVector2, const string& matchingType);
int createFeatureVectorCSV(const string& csvFilePath, const string& matchingType, const string& imgDir);
double cosineDistance(const Mat& v1, const Mat& v2);
vector<double> createLawsFeatureVector(vector<Mat>& filtered_images);
vector<Mat> applyLawsFilter(const Mat& image, const vector<Mat>& kernels);
vector<Mat> generateLawsKernels();
vector<double> generate1DHistogram(Mat& image, int num_bins);
bool detectBanana(const Mat& image);
int sobelX3x3(Mat &src, Mat &dst);
int sobelY3x3(Mat &src, Mat &dst);
int magnitude(Mat& sobelXFrame,Mat& sobelYFrame, Mat& dst);
double cosineDistance2(vector<double>& v1, vector<double>& v2);
Mat createGaborKernel(int size, double theta, double lambda, double sigma, double gamma);
vector<double> filterImageWithGabor(Mat& input, Mat& kernel);
bool detectBlueObjects(const Mat& image);


#endif // UTILITY_FUNCTIONS_H
