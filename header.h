#ifndef HEADER_H
#define HEADER_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>

#include <QCoreApplication>
#include<fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>
using namespace std;
using namespace cv;
using namespace cv::ml;
const string  files2[20] = { " Aa, Pieter van der (I)", " Buys, Jacobus",
                           " Cock, Hieronymus", "Dasrer, Albrecht", " Fokke, Simon",
                            " Staten van Holland en West-Friesland", " Visscher, Claes Jansz. (II)",
                            " Galle, Philips", " Galle, Theodoor", " Gheyn, Jacob de (II)"," Goltzius, Hendrick", " Hogenberg, Frans",
                            " Hooghe, Romeyn de", " Lodewijk XIV (koning van Frankrijk)", " Luyken, Jan"," Meissener Porzellan Manufaktur", " Passe, Crispijn van de (I)",
                            " Picart, Bernard", " Pronk, Cornelis", " Rembrandt Harmensz. van Rijn" };
const string  files[10] = {  "Buys, Jacobus",
                             "Fokke, Simon",
                             "Visscher, Claes Jansz. (II)",
                            "Galle, Philips",  "Hogenberg, Frans",
                            "Hooghe, Romeyn de", "Lodewijk XIV (koning van Frankrijk)","Meissener Porzellan Manufaktur",
                            "Picart, Bernard", "Rembrandt Harmensz. van Rijn" };
vector<Mat>SIFTDescriptors;
vector<vector<KeyPoint>>SIFTKeypoints;
vector<int> test_labels;
vector<Mat> test_imgs;
float how_many = 0.75;
vector<int> train_labels;
float countt = 0;
float accuracy = 0;
Mat testR;

void bag_of_words_create_dictionary(string files[]);
vector<Mat> bow_representation_extract_train(vector<int> *train_labels,vector<Mat> *SIFTDescriptors);
vector<Mat> bow_representation_extract_test(vector<int> *test_labels,Ptr<SVM> svm);
Ptr<SVM> SVMtrain(Mat trainMat, vector<int> trainLabels);
void ConvertVectortoMatrix(vector<Mat> &ipHOG, Mat & opMat);
float SVMevaluate(Mat testResponse, float count, vector<int> testLabels);
#endif // HEADER_H
