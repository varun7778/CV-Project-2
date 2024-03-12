#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "utility_functions.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << "<databaseDir> <MatchingType>" << std::endl;
        return 1;
    }

    // Specify the directory containing images
    string directoryPath = argv[1];
    string MatchingType = argv[2];

    // Specify the output CSV file
    string csvFilePath = "output.csv";
    
    // Open the CSV file for writing
    ofstream outputFile(csvFilePath);

    if (!outputFile.is_open()) {
        cerr << "Error opening CSV file for writing." << endl;
        return 1;
    }

    outputFile << MatchingType << endl;
    
    cv::String folderPath(directoryPath);
    std::vector<cv::String> files;
    cv::glob(folderPath, files);

    for (const auto& filePath : files) {

        cv::Mat image = cv::imread(filePath);
        if (image.empty()) {
            cerr << "Error reading image: " << filePath << endl;
            continue;
        }

        vector<double> featureVector;        
        if(MatchingType == "multihistogram")
        {
            vector<double> wholeFeatureVector = computeFeatureVector(image, MatchingType);
            cv::Rect centerROI((image.cols - 7) / 2, (image.rows - 7) / 2, 7, 7);
            cv::Mat centerRegion = image(centerROI);
            vector<double> centerFeatureVector = computeFeatureVector(centerRegion, MatchingType);

            featureVector.reserve(wholeFeatureVector.size() + centerFeatureVector.size());
            featureVector.insert(featureVector.end(), wholeFeatureVector.begin(), wholeFeatureVector.end());
            featureVector.insert(featureVector.end(), centerFeatureVector.begin(), centerFeatureVector.end());
        }
        else{
            featureVector = computeFeatureVector(image, MatchingType);
        }

        // Ensure the feature vector is valid
        if (featureVector.empty()) {
            std::cerr << "Invalid feature vector for: " << filePath << std::endl;
            continue;
        }

        // Write the image name to the CSV file
        outputFile << filePath << ",";
        
        // Write the feature vector to the CSV file
        for (const auto& value : featureVector) {
            outputFile << value << ",";
        }

        outputFile << endl;
    }

    // Close the CSV file
    outputFile.close();

    cout << "Feature extraction and CSV creation completed successfully." << endl;

    return 0;
}
        
        