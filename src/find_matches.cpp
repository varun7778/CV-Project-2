/*
  Varun Anusheel, Lokesh Saipureddi
  Spring 2024
  CS 5330 Computer Vision

  Project 2: Content-Based Image Retrieval
*/
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "utility_functions.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc != 5 && argc != 4 && argc != 3 && argc != 2) {
        cerr << "Usage: " << argv[0] << " <databaseDir> <MatchingType> <targetImagePath> <N>" << endl;
        cerr << "Usage: " << argv[0] << " <ImageName> <N>" << endl;
        cerr << "Usage: " << argv[0] << " <databaseDir>" << endl;
        cerr << "Usage: " << argv[0] << " <databaseDir> <ImageName> findall" << endl;
        return 1;
    }

    if (argc == 5)
    {
        string directoryPath = argv[1];
        string MatchingType = argv[2];
        string targetImagePath = argv[3];
        int N = stoi(argv[4]);
        string csvFilePath = "D:/CV/Project_2/bin/Debug/output.csv";

        Mat targetImage = imread(targetImagePath);
        if (targetImage.empty()) {
            cerr << "Error reading target image: " << targetImagePath << endl;
            return 1;
        }

        vector<double> targetFeatureVector;
        vector<double> targetFeatureVectorLaws;
        if(MatchingType == "lawsfilter"){
            vector<Mat> kernels = generateLawsKernels();
            vector<Mat> filtered_images = applyLawsFilter(imread(targetImagePath, IMREAD_GRAYSCALE), kernels);
            targetFeatureVectorLaws = createLawsFeatureVector(filtered_images);
        } else if(MatchingType == "gabor"){
            targetImage.convertTo(targetImage, CV_64F, 1.0 / 255.0);
            int kernel_size = 15;
            double theta = CV_PI / 4;
            double lambda = 10.0;
            double sigma = 5.0;
            double gamma = 0.5;
            Mat gabor_kernel = createGaborKernel(kernel_size, theta, lambda, sigma, gamma);
            targetFeatureVector = filterImageWithGabor(targetImage, gabor_kernel);
        } else{
            targetFeatureVector = computeFeatureVector(targetImage,MatchingType);
        }

        if (!fileExists(csvFilePath)) {
            std::cout << "File does not exists.\n";
            int success = createFeatureVectorCSV(csvFilePath, MatchingType, directoryPath);
            if(success!=0)
            {
                cerr << "Error creating csv file."<< endl;
                return 1;
            }
        }

        double distance = 0.0;
        vector<pair<string, double>> imageDistances;

        if(MatchingType != "gabor")
        {
            ifstream inputFile(csvFilePath);
            if (!inputFile.is_open()) {
                cerr << "Error opening CSV file for reading." << endl;
                return 1;
            }

            string headerLine;
            getline(inputFile, headerLine);
            if(headerLine != MatchingType)
            {
                cerr << "MatchingType mismatch, delete the output.csv file and try again" << endl;
                return 1;
            }

            while (!inputFile.eof()) {
                string line;
                getline(inputFile, line);

                if (line.empty()) {
                    continue;
                }

                stringstream ss(line);
                string imageName, value;
                getline(ss, imageName, ',');

                vector<double> featureVector;
                while (getline(ss, value, ',')) {
                    featureVector.push_back(stod(value));
                }

                if (MatchingType == "baseline") {
                    distance = computeDistance(targetFeatureVector, featureVector, MatchingType);
                } else if(MatchingType == "histogram") {

                    distance = computeIntersection(targetFeatureVector, featureVector)*-1;
                } else if(MatchingType == "multihistogram"){

                    double distance1, distance2;
                    std::vector<double> wholeFeatureVector(featureVector.begin(), featureVector.begin() + 256);
                    std::vector<double> centerFeatureVector(featureVector.begin() + 256, featureVector.end());
                                
                    cv::Rect centerROI((targetImage.cols - 7) / 2, (targetImage.rows - 7) / 2, 7, 7);
                    cv::Mat targetCenterRegion = targetImage(centerROI);
                    vector<double> TargetCenterFeatureVector = computeFeatureVector(targetCenterRegion, MatchingType);

                    distance1 = computeIntersection(targetFeatureVector, wholeFeatureVector)*-1;
                    distance2 = computeIntersection(TargetCenterFeatureVector, centerFeatureVector)*-1;


                    distance = (distance1*0.6) + (distance2*0.4);
                } else if(MatchingType == "lawsfilter"){

                    vector<double> lawsFeatureVector;
                    vector<double> wholeFeatureVector;

                    lawsFeatureVector.reserve(400);
                    wholeFeatureVector.reserve(256);
                    lawsFeatureVector.insert(lawsFeatureVector.end(), featureVector.begin(), featureVector.begin() + 400);
                    wholeFeatureVector.insert(wholeFeatureVector.end(), featureVector.begin() + 400, featureVector.end());


                    double distance1, distance2;
                    distance1 = computeDistance(targetFeatureVectorLaws, lawsFeatureVector, MatchingType);
                    distance2 = computeIntersection(targetFeatureVector,wholeFeatureVector)*-1;

                    distance = (distance1*0.75) + (distance2*0.25);

                } else if(MatchingType == "hog"){

                    vector<double> wholeFeatureVector(featureVector.begin(), featureVector.begin() + 256);
                    vector<double> hogsFeatureVector(featureVector.begin() + 256, featureVector.end());

                    
                    vector<double> wholeTargetFeatureVector = computeFeatureVector(targetImage,"histogram");
                    double distance1, distance2;
                    distance1 = cosineDistance2(targetFeatureVector, hogsFeatureVector);
                    distance2 = computeIntersection(wholeTargetFeatureVector,wholeFeatureVector)*-1;

                    distance = (distance1*0.5) + (distance2*0.5);

                } else if(MatchingType == "texture"){

                    Mat grayImage = imread(imageName,IMREAD_GRAYSCALE);
                    Mat sobelX, sobelY;
                    sobelX3x3(grayImage, sobelX);
                    sobelY3x3(grayImage, sobelY);
                    Mat mag;
                    magnitude(sobelX, sobelY, mag);
                    vector<double> imgSobelFeatureVector = generate1DHistogram(mag, 16);

                    vector<double> colorFeatureVector(featureVector.begin(), featureVector.begin() + 256);
                    vector<double> sobelFeatureVector(featureVector.begin() + 256, featureVector.end());

                    double distance1 = cosineDistance2(imgSobelFeatureVector, sobelFeatureVector);
                    double distance2 = cosineDistance2(targetFeatureVector, featureVector);
                    
                    distance = 0.5 * distance1 + 0.5 * distance2;    

                } else {
                    cerr << "Unknown matching type: " << MatchingType << endl;
                    return 1;
                }

                imageDistances.emplace_back(imageName, distance);
            }

            inputFile.close();
        }

        else{
            ifstream inputFile(csvFilePath);
            if (!inputFile.is_open()) {
                cerr << "Error opening CSV file for reading." << endl;
                return 1;
            }

            string headerLine;
            getline(inputFile, headerLine);
            if(headerLine != MatchingType)
            {
                cerr << "MatchingType mismatch, delete the output.csv file and try again" << endl;
                return 1;
            }

            vector<string> filenames;
            vector<vector<double>> featureVectors;
            while (!inputFile.eof()) {
                string line;
                getline(inputFile, line);
                if (line.empty()) {
                    continue;
                }
                stringstream ss(line);
                string imageName, value;
                getline(ss, imageName, ',');
                filenames.push_back(imageName);
                vector<double> featureVector;
                while (getline(ss, value, ',')) {
                    featureVector.push_back(stod(value));
                }
                featureVectors.push_back(featureVector);
            }
            inputFile.close();

            ifstream inputFile1("D:/CV/Project_2/ResNet18_olym.csv");
            if (!inputFile1.is_open()) {
                cerr << "Error opening the CSV file." << endl;
            }
            vector<string> filenames1;
            vector<vector<double>> featureVectors1;
            while (!inputFile1.eof()) {
                string line;
                getline(inputFile1, line);
                if (line.empty()) {
                    continue;
                }
                stringstream ss(line);
                string imageName, value;
                getline(ss, imageName, ',');
                filenames1.push_back("D:/CV/Project_2/olympus/olympus/" + imageName);
                vector<double> featureVector1;
                while (getline(ss, value, ',')) {
                    featureVector1.push_back(stod(value));
                }
                featureVectors1.push_back(featureVector1);
            }
            inputFile1.close();
            string targetFilename = targetImagePath;
            vector<double> targetVector;
            for (size_t i = 0; i < filenames1.size(); ++i) {
                if (filenames1[i] == targetFilename) {
                    targetVector = featureVectors1[i];
                    break;
                }
            }

            for (size_t i = 0; i < filenames.size(); ++i) {
                if (filenames[i] != targetImagePath) {
                    double distance1 = computeDistance(targetFeatureVector, featureVectors[i],MatchingType);
                    double distance2 = computeDistance(targetVector, featureVectors1[i],MatchingType);
                    double distance = 0.3*distance1 + 0.7*distance2;
                    imageDistances.push_back({filenames[i], distance});
                }
            }

        }


        sort(imageDistances.begin(), imageDistances.end(),
            [](const pair<string, double>& a, const pair<string, double>& b) {
                return a.second < b.second;
            });

        cout << "Top " << N << " matches:" << endl;

        for (int i = 0; i < min(N, static_cast<int>(imageDistances.size())); ++i) {
            cout << "Image Name: " << imageDistances[i].first << ", Distance: " << imageDistances[i].second << endl;

            Mat resultImage = imread(imageDistances[i].first);
            if (!resultImage.empty()) {
                namedWindow("Result Image", WINDOW_NORMAL);
                imshow("Result Image", resultImage);
                waitKey(0);
            } else {
                cerr << "Error: Unable to open image: " << imageDistances[i].first << endl;
            }
        }
        
    } else if(argc == 3)
    {
        string targetFilename = argv[1];
        int N = stoi(argv[2]);
        ifstream inputFile("D:/CV/Project_2/ResNet18_olym.csv");

        if (!inputFile.is_open()) {
            cerr << "Error opening the CSV file." << endl;
            return -1;
        }
        
        vector<string> filenames;
        vector<Mat> featureVectors;
        while (!inputFile.eof()) {
            string line;
            getline(inputFile, line);
            if (line.empty()) {
                continue;
            }
            stringstream ss(line);
            string imageName, value;
            getline(ss, imageName, ',');
            filenames.push_back(imageName);
            vector<double> featureVector;
            while (getline(ss, value, ',')) {
                featureVector.push_back(stod(value));
            }
            featureVectors.push_back(Mat(featureVector).clone());
        }
        Mat targetVector;

        for (size_t i = 0; i < filenames.size(); ++i) {
            if (filenames[i] == targetFilename) {
                targetVector = featureVectors[i];
                break;
            }
        }

        vector<pair<string, double>> results;

        for (size_t i = 0; i < filenames.size(); ++i) {
            if (filenames[i] != targetFilename) {
                double distance = cosineDistance(targetVector, featureVectors[i]);
                results.push_back({filenames[i], distance});
            }
        }

        sort(results.begin(), results.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
                return a.second < b.second;
            });
        
        cout << "Top " << N << " matches:" << endl;

        for (int i = 0; i < min(N, static_cast<int>(results.size())); ++i) {
            cout << "Image Name: " << results[i].first << ", Distance: " << results[i].second << endl;
            }
    } else if(argc == 2){
        string targetFilename = argv[1];
        string directoryPath = "D:/CV/Project_2/olympus/olympus";
        cv::String folderPath(directoryPath);
        std::vector<cv::String> files;
        cv::glob(folderPath, files);

        for (const auto& filePath : files) {
            cv::Mat image = cv::imread(filePath);
            if (image.empty()) {
                cerr << "Error reading image: " << filePath << endl;
                continue;
            }
            bool bananaDetected = detectBanana(image);
        }
    } else if(argc == 4)
    {
        string targetDir = argv[1];
        string targetFilename = argv[2];
        string findall = argv[3];
        ifstream inputFile("D:/CV/Project_2/ResNet18_olym.csv");

        if (!inputFile.is_open()) {
            cerr << "Error opening the CSV file." << endl;
            return -1;
        }
        
        vector<string> filenames;
        vector<Mat> featureVectors;
        while (!inputFile.eof()) {
            string line;
            getline(inputFile, line);
            if (line.empty()) {
                continue;
            }
            stringstream ss(line);
            string imageName, value;
            getline(ss, imageName, ',');
            filenames.push_back(imageName);
            vector<double> featureVector;
            while (getline(ss, value, ',')) {
                featureVector.push_back(stod(value));
            }
            featureVectors.push_back(Mat(featureVector).clone());
        }
        Mat targetVector;

        for (size_t i = 0; i < filenames.size(); ++i) {
            if (filenames[i] == targetFilename) {
                targetVector = featureVectors[i];
                break;
            }
        }

        vector<pair<string, double>> results;

        for (size_t i = 0; i < filenames.size(); ++i) {
            if (filenames[i] != targetFilename) {
                double distance = cosineDistance(targetVector, featureVectors[i]);
                results.push_back({filenames[i], distance});
            }
        }

        sort(results.begin(), results.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
                return a.second < b.second;
            });

        vector<string> blueTrashCanImages;
        for (const auto& entry : results) {
            Mat image = imread("D:/CV/Project_2/olympus/olympus/" + entry.first);
            if(entry.second >= 0.292)
            {
                continue;
            }
            if (detectBlueObjects(image)) {
                imshow("Trash Regions", image);
                waitKey(0);
                blueTrashCanImages.push_back(entry.first);
                cout << "Trashcan detected at "<< entry.first <<": "<< entry.second << endl;
            }
        }   
        
    } 
    return 0;
}
