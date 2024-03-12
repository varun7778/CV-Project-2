/*
  Varun Anusheel, Lokesh Saipureddi
  Spring 2024
  CS 5330 Computer Vision

  Project 2: Content-Based Image Retrieval. This file contains all the helper functions used in  the main program
*/

#include "utility_functions.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


/*
  Computes the feature vector of an image based on the specified matching type.
  If matchingType is "baseline", extracts a 7x7 region around the center of the image.
  If matchingType is "histogram", "texture", or "lawsfilter", computes the corresponding histogram-based feature vector.
  If matchingType is "hog", computes the Histogram of Oriented Gradients (HOG) feature vector.
*/
vector<double> computeFeatureVector(Mat& image, const string& matchingType) {
    vector<double> featureVector;
    if (matchingType == "baseline") {
        Rect roi((image.cols - 7) / 2, (image.rows - 7) / 2, 7, 7);
        Mat regionOfInterest = image(roi);

        featureVector.reserve(regionOfInterest.rows * regionOfInterest.cols);

        for (int i = 0; i < regionOfInterest.rows; ++i) {
            for (int j = 0; j < regionOfInterest.cols; ++j) {
                featureVector.push_back(regionOfInterest.at<uchar>(i, j));
            }
        }

        return featureVector;
    } else if (matchingType.find("histogram") != std::string::npos || matchingType == "texture" || matchingType == "lawsfilter") {
        Mat dst;
        Mat hist;
        float max;
        const int histsize = 16;
        hist = Mat::zeros(Size(histsize, histsize), CV_32FC1);
        max = 0;
        for (int i = 0; i < image.rows; i++) {
        Vec3b * ptr = image.ptr < Vec3b > (i);
        for (int j = 0; j < image.cols; j++) {
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0;
            float r = R / divisor;
            float g = G / divisor;
            int rindex = (int)(r * (histsize - 1) + 0.5);
            int gindex = (int)(g * (histsize - 1) + 0.5);
            hist.at < float > (rindex, gindex) ++;
            float newvalue = hist.at < float > (rindex, gindex);
            max = newvalue > max ? newvalue : max;
        }
        }
        hist /= (image.rows * image.cols); 

        featureVector.reserve(16 * 16);
        for (int h = 0; h < hist.rows; ++h) {
            for (int s = 0; s < hist.cols; ++s) {
                featureVector.push_back(hist.at<float>(h, s));
            }
        }

        return featureVector;
    } else if(matchingType == "hog")
    {
        resize(image, image, Size(64,128) );
        Mat img;
        cvtColor(image, img, COLOR_BGR2GRAY);
        vector<float> features;
        vector<Point> locations;

        HOGDescriptor *hog = new HOGDescriptor();
        hog->compute(img,features,Size(32,32), Size(0,0),locations);
        cout<<features.size()<<endl;

        Mat Hogfeat;
        Hogfeat.create(features.size(),1,CV_32FC1);
        for(int i=0; i<features.size(); i++)
            Hogfeat.at<float>(i,0)=features.at(i);

        
        for(int i=0; i<Hogfeat.rows; i++) {
            featureVector.push_back(static_cast<double>(Hogfeat.at<float>(i, 0)));
        }
        return featureVector;
    } else {
        cerr << "Unknown matching type: " << matchingType << endl;
        return vector<double>();
    }
}


/*
  Computes the intersection between two histograms represented by vectors hist1 and hist2.
*/
double computeIntersection(vector<double>& hist1, vector<double>& hist2) {
    if (hist1.size() != hist2.size()) {
        cerr << "Error: Histograms must be of the same size for intersection computation." << endl;
        return -1.0;
    }

    double intersection = 0.0;
    for (size_t i = 0; i < hist1.size(); ++i) {
        intersection += min(hist1[i], hist2[i]);
    }
    return intersection;
}

/*
  Visualizes a histogram represented by a matrix 'hist'.
*/
void visualizeHist(Mat& hist)
{
    Mat dst;
    const int histsize = 16;
    dst.create(hist.size(), CV_8UC3);
        for (int i = 0; i < hist.rows; i++) {
        Vec3b * ptr = dst.ptr < Vec3b > (i);
        float * hptr = hist.ptr < float > (i);
        for (int j = 0; j < hist.cols; j++) {
            if (i + j > hist.rows) {
            ptr[j] = Vec3b(200, 120, 60);
            continue;
            }
            float rcolor = (float) i / histsize;
            float gcolor = (float) j / histsize;
            float bcolor = 1 - (rcolor + gcolor);
            ptr[j][0] = hptr[j] > 0 ? hptr[j] * 128 + 128 * bcolor : 0;
            ptr[j][1] = hptr[j] > 0 ? hptr[j] * 128 + 128 * gcolor : 0;
            ptr[j][2] = hptr[j] > 0 ? hptr[j] * 128 + 128 * rcolor : 0;
        }
        }
        imshow("Histogram", dst);
        waitKey(0);
}

/*
  Computes the distance between two feature vectors based on the specified matching type.
  For "baseline" and "lawsfilter", uses Euclidean distance.
  For "hog", counts the number of non-matching elements.
*/
double computeDistance(const vector<double>& featureVector1, const vector<double>& featureVector2, const string& matchingType) {
    double distance = 0.0;
    if(matchingType == "baseline" || matchingType == "gabor")
    {
        for (size_t i = 0; i < featureVector1.size(); ++i) {
            distance += pow(featureVector1[i] - featureVector2[i], 2);
        }
    } else if(matchingType == "lawsfilter")
    {
        for (size_t i = 0; i < featureVector1.size(); ++i) {
            distance += pow(featureVector1[i] - featureVector2[i], 2);
        }
        distance = sqrt(distance);
    } else if(matchingType == "hog"){
        for (size_t i = 0; i < featureVector1.size(); ++i) {
            if (featureVector1[i] != featureVector2[i]) {
                ++distance;
            }
        }
    }

    return distance;
}

/*
  Checks if a file exists given its filename.
*/
bool fileExists(const string& filename) {
    ifstream file(filename);
    return file.good();
}

/*
  Creates a CSV file containing feature vectors extracted from images in the specified directory.
  Supports various matching types including "multihistogram", "lawsfilter", "hog", and "texture".
*/
int createFeatureVectorCSV(const string& csvFilePath, const string& matchingType, const string& imgDir){
    ofstream outputFile(csvFilePath);

    if (!outputFile.is_open()) {
        cerr << "Error opening CSV file for writing." << endl;
        return 1;
    }

    outputFile << matchingType << endl;
    
    cv::String folderPath(imgDir);
    std::vector<cv::String> files;
    cv::glob(folderPath, files);

    for (const auto& filePath : files) {

        cv::Mat image = cv::imread(filePath);
        if (image.empty()) {
            cerr << "Error reading image: " << filePath << endl;
            continue;
        }

        vector<double> featureVector;        
        if(matchingType == "multihistogram")
        {
            vector<double> wholeFeatureVector = computeFeatureVector(image, matchingType);
            cv::Rect centerROI((image.cols - 7) / 2, (image.rows - 7) / 2, 7, 7);
            cv::Mat centerRegion = image(centerROI);
            vector<double> centerFeatureVector = computeFeatureVector(centerRegion, matchingType);

            featureVector.reserve(wholeFeatureVector.size() + centerFeatureVector.size());
            featureVector.insert(featureVector.end(), wholeFeatureVector.begin(), wholeFeatureVector.end());
            featureVector.insert(featureVector.end(), centerFeatureVector.begin(), centerFeatureVector.end());
        } else if(matchingType == "lawsfilter"){
            Mat grayImage;
            cvtColor(image, grayImage, COLOR_BGR2GRAY);

            vector<Mat> kernels = generateLawsKernels();
            vector<Mat> filtered_images = applyLawsFilter(grayImage, kernels);
            vector<double> lawsFeatureVector = createLawsFeatureVector(filtered_images);

            vector<double> wholeFeatureVector = computeFeatureVector(image, "histogram");
            featureVector.reserve(lawsFeatureVector.size() + wholeFeatureVector.size());
            featureVector.insert(featureVector.end(), lawsFeatureVector.begin(), lawsFeatureVector.end());
            featureVector.insert(featureVector.end(), wholeFeatureVector.begin(), wholeFeatureVector.end());
        } else if(matchingType == "hog"){

            vector<double> hogsFeatureVector = computeFeatureVector(image, matchingType);
            vector<double> wholeFeatureVector = computeFeatureVector(image, "histogram");
            featureVector.reserve(hogsFeatureVector.size() + wholeFeatureVector.size());

            featureVector.insert(featureVector.end(), wholeFeatureVector.begin(), wholeFeatureVector.end());
            featureVector.insert(featureVector.end(), hogsFeatureVector.begin(), hogsFeatureVector.end());
        } else if(matchingType == "texture"){
            Mat grayImage;
            cvtColor(image, grayImage, COLOR_BGR2GRAY);
            Mat sobelX, sobelY;
            sobelX3x3(grayImage, sobelX);
            sobelY3x3(grayImage, sobelY);
            Mat mag;
            magnitude(sobelX, sobelY, mag);
            vector<double> sobelFeatureVector = generate1DHistogram(mag, 16);
            vector<double> colorFeatureVector = computeFeatureVector(image, matchingType);
            featureVector.reserve(sobelFeatureVector.size() + colorFeatureVector.size());
            featureVector.insert(featureVector.end(), colorFeatureVector.begin(), colorFeatureVector.end());
            featureVector.insert(featureVector.end(), sobelFeatureVector.begin(), sobelFeatureVector.end());
        } else if(matchingType == "gabor"){

            int kernel_size = 15;
            double theta = CV_PI / 4;
            double lambda = 10.0;
            double sigma = 5.0;
            double gamma = 0.5;
            image.convertTo(image, CV_64F, 1.0 / 255.0);
            Mat gabor_kernel = createGaborKernel(kernel_size, theta, lambda, sigma, gamma);
            featureVector = filterImageWithGabor(image, gabor_kernel);

        } else{
            featureVector = computeFeatureVector(image, matchingType);
        }

        if (featureVector.empty()) {
            std::cerr << "Invalid feature vector for: " << filePath << std::endl;
            continue;
        }

        outputFile << filePath << ",";
        
        for (const auto& value : featureVector) {
            outputFile << value << ",";
        }

        outputFile << endl;
    }

    outputFile.close();

    cout << "Feature extraction and CSV creation completed successfully." << endl;

    return 0;
}

/*
  Computes the cosine distance between two vectors v1 and v2.
*/
double cosineDistance(const Mat& v1, const Mat& v2) {
    double dotProduct = v1.dot(v2);
    double normV1 = norm(v1);
    double normV2 = norm(v2);
    return 1.0 - (dotProduct / (normV1 * normV2));
}

/*
  Generates a set of 5x5 2D convolution kernels for Laws' texture energy measures.
*/
vector<Mat> generateLawsKernels() {
    const int kernel_size = 5;
    const int num_filters = 9;
    const int num_bins = 16;
    vector<Mat> kernels;

    int laws_kernels[5][5] = {{1, 4, 6, 4, 1},
                               {-1, -2, 0, 2, 1},
                               {-1, 0, 2, 0, -1},
                               {1, -4, 6, -4, 1},
                               {-1, 2, 0, -2, 1}};

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
            for (int m = 0; m < kernel_size; ++m) {
                for (int n = 0; n < kernel_size; ++n) {
                    kernel.at<float>(m, n) = laws_kernels[i][m] * laws_kernels[j][n];
                }
            }
            kernels.push_back(kernel);
        }
    }
    return kernels;
}

/*
  Applies Laws' texture energy measures filters to the input image using the provided kernels.
*/
vector<Mat> applyLawsFilter(const Mat& image, const vector<Mat>& kernels) {
    vector<Mat> filtered_images;

    for (const auto& kernel : kernels) {
        Mat filtered;
        filter2D(image, filtered, CV_32F, kernel);
        filtered_images.push_back(filtered);
    }
    return filtered_images;
}

/*
  Creates a feature vector from the filtered images obtained after applying Laws' filters.
*/
vector<double> createLawsFeatureVector(vector<Mat>& filtered_images) {
    const int num_filters = 9;
    const int num_bins = 16;
    vector<double> featureVector;

    for (auto& filtered_image : filtered_images) {
        vector<double> feature_vector = generate1DHistogram(filtered_image, num_bins);
        featureVector.insert(featureVector.end(), feature_vector.begin(), feature_vector.end());
    }

    return featureVector;
}

/*
  Generates a 1D histogram from the input image with the specified number of bins.
*/
vector<double> generate1DHistogram(Mat& image, int num_bins)
{
    Mat hist;
    float max = 0;

    hist = Mat::zeros(Size(num_bins, 1), CV_32FC1);

    for (int i = 0; i < image.rows; i++) {
        Vec3b * ptr = image.ptr<Vec3b>(i);
        for (int j = 0; j < image.cols; j++) {
            double intensity = ptr[j][0];
            int index = static_cast<int>(intensity * (num_bins - 1) / 255);
            hist.at<float>(0, index)++;
            float newvalue = hist.at<float>(0, index);
            max = max < newvalue ? newvalue : max;
        }
    }

    hist /= (image.rows * image.cols);
    vector<double> histVec;
    for (int i = 0; i < hist.cols; i++) {
        histVec.push_back(hist.at<float>(0, i));
    }
    return histVec;
}

/*
  Detects banana regions in the input image based on color and size constraints.
*/
bool detectBanana(const Mat& image) {
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    Scalar lowerBound(20, 100, 100);
    Scalar upperBound(30, 255, 255);

    Mat mask;
    inRange(hsvImage, lowerBound, upperBound, mask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> boundingBoxes;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > 1500 && area < 5000) {
            RotatedRect boundingBox = minAreaRect(contour);
            Rect bbox = minAreaRect(contour).boundingRect();
            boundingBoxes.push_back(bbox);
            double aspectRatio = boundingBox.size.width / boundingBox.size.height;
            if (aspectRatio > 1.0 && aspectRatio < 4.0) {
                Rect bbox = boundingBox.boundingRect();
                boundingBoxes.push_back(bbox);
            }
        }
    }

    if (boundingBoxes.empty()) {
        return false;
    }

    for (const auto& bbox : boundingBoxes) {
        rectangle(image, bbox, Scalar(0, 255, 0), 2);
    }

    imshow("Banana Regions", image);
    waitKey(0);
    return true;
}

bool detectBlueObjects(const Mat& image) {
    Mat blurred;
    GaussianBlur(image, blurred, Size(11, 11), 0);

    Mat hsv;
    cvtColor(blurred, hsv, COLOR_BGR2HSV);
    
    // Scalar lowerBlue = Scalar(100, 50, 50);
    // Scalar upperBlue = Scalar(140, 255, 255);

    Scalar lowerBlue = Scalar(90, 50, 70);
    Scalar upperBlue = Scalar(130, 255, 255);

    Mat mask;
    inRange(hsv, lowerBlue, upperBlue, mask);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if(contours.size() >0)
    {
        return true;
    }
    return false;
}

/*
  Applies a 3x3 Sobel filter in the X direction to the input image.
*/
int sobelX3x3(Mat &src, Mat &dst) {
    Mat src_temp;
    Mat tempImage;

    src.convertTo(src_temp, CV_16SC3);
    src.convertTo(tempImage, CV_16SC3);

    dst.create(src.rows, src.cols, CV_16SC3);

    double kernel[3] = {1, 2, 1}; 

    for (int i = 0; i < src.rows; i++) {
        Vec3s *row_ptr = src_temp.ptr<Vec3s>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < src.channels(); k++) {
                double sum = 0;
                for (int m = 0; m < 3; m++) {
                    sum += kernel[m] * row_ptr[j - 1 + m][k];
                }
                tempImage.at<Vec3s>(i, j - 1)[k] = sum;
            }
        }
    }

    double kernel1[3] = {-1, 0, 1};
    
    for (int i = 1; i < src.cols - 1; i++) {
        for (int j = 0; j < src.rows; j++) {
            Vec3s *col_ptr = tempImage.ptr<Vec3s>(j);
            for (int k = 0; k < src.channels(); k++) {
                double sum = 0;
                
                for (int m = 0; m < 3; m++) {
                    sum += kernel1[m] * col_ptr[i - 1 + m][k];
                }
                
                dst.at<Vec3s>(j, i - 1)[k] = sum;
            }
        }
    }

    return 0;
}

/*
  Applies a 3x3 Sobel filter in the Y direction to the input image.
*/
int sobelY3x3(Mat &src, Mat &dst) {
    Mat src_temp;
    Mat tempImage;

    src.convertTo(src_temp, CV_16SC3);
    src.convertTo(tempImage, CV_16SC3);

    dst.create(src.rows, src.cols, CV_16SC3);

    double kernel[3] = {-1, 0, 1};

    for(int i = 1; i < src.rows - 1; i++) {
        for(int j = 0; j < src.cols; j++) {
            for(int k = 0; k < 3; k++) {
                double sum = 0;
                for (int m = 0; m < 3; m++) {
                    sum += kernel[m] * src_temp.at<Vec3s>(i - 1 + m, j)[k];
                }
                tempImage.at<Vec3s>(i - 1, j)[k] = sum;
            }
        }
    }

    double kernel1[3] = {1, 2, 1};

    for (int i = 0; i < src.rows; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < src.channels(); k++) {
                double sum = 0;
                for (int m = 0; m < 3; m++) {
                    sum += kernel1[m] * tempImage.at<Vec3s>(i, j - 1 + m)[k];
                }
                dst.at<Vec3s>(i, j - 1)[k] = sum;
            }
        }
    }

    return 0;
}


/*
  Computes the magnitude of gradient images obtained using Sobel operators in both X and Y directions.
*/
int magnitude(Mat& sobelXFrame,Mat& sobelYFrame, Mat& dst) {
    if (sobelXFrame.empty() || sobelYFrame.empty()) {
        return -1;
    }

    dst.create(sobelXFrame.size(), CV_8UC3);

    for (int i = 0; i < sobelXFrame.rows; ++i) {
        for (int j = 0; j < sobelXFrame.cols; ++j) {
            for (int k = 0; k < 3; ++k) {
                float mag = std::sqrt(static_cast<float>(sobelXFrame.at<cv::Vec3s>(i, j)[k] * sobelXFrame.at<cv::Vec3s>(i, j)[k] +
                                                        sobelYFrame.at<cv::Vec3s>(i, j)[k] * sobelYFrame.at<cv::Vec3s>(i, j)[k]));

                dst.at<cv::Vec3s>(i, j)[k] = static_cast<uchar>(std::min(255.0f, mag));
            }
        }
    }
    return 0;
}


/*
  Computes the cosine distance between two vectors v1 and v2.
*/
double cosineDistance2(vector<double>& v1, vector<double>& v2) {
   if (v1.size() != v2.size() || v1.empty()) {
        std::cerr << "Vector dimensions mismatch or empty vectors." << std::endl;
        return 0.0;
    }
    double dotProduct = 0.0;
    double normV1 = 0.0;
    double normV2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        normV1 += v1[i] * v1[i];
        normV2 += v2[i] * v2[i];
    }
    double distance = 1.0 - (dotProduct / (std::sqrt(normV1) * std::sqrt(normV2)));
    return distance;
}

/*
 Creates a Gabor kernel with given parameters for image filtering.
*/

Mat createGaborKernel(int size, double theta, double lambda, double sigma, double gamma) {
    Mat kernel(size, size, CV_64F);
    double sigma_x = sigma;
    double sigma_y = sigma / gamma;
    double half_size = size / 2;
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    for (int x = -half_size; x <= half_size; ++x) {
        for (int y = -half_size; y <= half_size; ++y) {
            double x_theta = x * cos_theta + y * sin_theta;
            double y_theta = -x * sin_theta + y * cos_theta;

            double term1 = exp(-(x_theta * x_theta + y_theta * y_theta) / (2 * sigma * sigma));
            double term2 = cos(2 * CV_PI * x_theta / lambda);

            kernel.at<double>(x + half_size, y + half_size) = term1 * term2;
        }
    }

    normalize(kernel, kernel, 1.0, 0.0, NORM_L1);

    return kernel;
}

/*
  Filters an image using a given Gabor kernel and returns the filtered image histogram.
*/
vector<double> filterImageWithGabor(Mat& input, Mat& kernel) {
    Mat filtered_image(input.size(), CV_64F, Scalar(0));
    vector<double> fi;
    int kernel_size = kernel.rows;
    int half_size = kernel_size / 2;

    for (int x = half_size; x < input.rows - half_size; ++x) {
        for (int y = half_size; y < input.cols - half_size; ++y) {
            double sum = 0.0;

            for (int i = -half_size; i <= half_size; ++i) {
                for (int j = -half_size; j <= half_size; ++j) {
                    sum += input.at<double>(x + i, y + j) * kernel.at<double>(i + half_size, j + half_size);
                }
            }

            filtered_image.at<double>(x, y) = sum;
        }
    }
    normalize(filtered_image, filtered_image, 0, 255, NORM_MINMAX);
    fi = generate1DHistogram(filtered_image,16);
    return fi;
}