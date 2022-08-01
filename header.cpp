// Function codes for EE_576 Project 4
// Mehmet Yiğit Avcı
// Bogazici University, 2022

#include<header.h>

// function to create BoW Dictionary
// it needs to be only called once to create dictionary
// it uses SIFT descriptors
void bag_of_words_create_dictionary()
{
      float how_many = 0.75;

      vector<KeyPoint> keypoints;
      Mat descriptors;
      Mat train_set_features;

       //The SIFT feature extractor and descriptor
       auto detector = cv::SiftFeatureDetector::create();
       auto extractor = cv::SiftDescriptorExtractor::create();
       int size=0;
       for (int i = 0; i < 10; ++i)
            {
                cout << files[i] << endl;

                std::string folder = "D:/ee58j_final_project/art_imgs_10/"+files[i]+"/*.jpg";
                std::vector<std::string> filenames;
                cv::glob(folder, filenames);
                random_shuffle(filenames.begin(), filenames.end());
                size_t N = filenames.size() * how_many;
                size=size+N;
                for (size_t k = 0; k < N; ++k)
                {
                    // Load and show image
                    cv::Mat img = cv::imread(filenames[k]);
                    detector->detect(img, keypoints);
                    //compute the descriptors for each keypoint
                    extractor->compute(img, keypoints, descriptors);
                    //put the all feature descriptors in a single Mat object
                    train_set_features.push_back(descriptors);
                }

            }

              //define Term Criteria
              TermCriteria tc(TermCriteria::MAX_ITER|TermCriteria::EPS,100,0.001);
              //retries number
              int retries=1;
              //necessary flags
              int flags=KMEANS_RANDOM_CENTERS;
              //BoW trainer
              BOWKMeansTrainer bowTrainer(50,tc,retries,flags);
              //clustering feature vectors
              Mat dictionary=bowTrainer.cluster(train_set_features);
              //saving the dictionary
              FileStorage fs("D:/ee58j_final_project/dictionary.yml", FileStorage::WRITE);
              fs << "vocabulary" << dictionary;
              fs.release();

}

// this function finds the BoW representation of the training data and their labels
// to feed the SVM classifier
vector<Mat> bow_representation_extract_train(vector<int> *train_labels,vector<Mat> *SIFTDescriptors,vector<vector<KeyPoint>> *SIFTKeypoints)
{

    vector<Mat> bow_representation;

    //load dictionary
    Mat dictionary;
    FileStorage fs("D:/ee58j_final_project/dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();

    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

    auto detector = cv::SiftFeatureDetector::create();
    auto extractor = cv::SiftDescriptorExtractor::create();
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Dictionary is set with the vocabulary we created
    bowDE.setVocabulary(dictionary);

    cout <<" Obtaining BoW representation of train set..."<< endl;
    for (int i = 0; i < 10; ++i)
    {
        cout << files[i] << endl;

        std::string folder = "D:/ee58j_final_project/art_imgs_10/"+files[i]+"/*.jpg";
        std::vector<std::string> filenames;
        cv::glob(folder, filenames);
        random_shuffle(filenames.begin(), filenames.end());
        size_t N = filenames.size()*0.75 ;

        for (size_t c = 0; c <N; ++c)
        {
            //read the image
            Mat img=imread(filenames[c]);
            vector<KeyPoint> keypoints;
            detector->detect(img,keypoints);
            Mat bowDescriptor;
            //extract BoW  descriptor from given image
            bowDE.compute(img,keypoints,bowDescriptor);
            // create the BoW representation to feed SVM classifier
            bow_representation.push_back(bowDescriptor);
            // labels of the data
            train_labels->push_back(i);
            if (i==0)
            {
                Mat siftdesc;
                extractor->compute(img,keypoints,siftdesc);
                SIFTDescriptors->push_back(siftdesc);
                SIFTKeypoints->push_back(keypoints);

            }

        }
    }
    return bow_representation;
}

// this function finds the BoW representation of the test data and their labels
vector<Mat> bow_representation_extract_test(vector<int> *test_labels,Ptr<SVM> svm)
{

    vector<Mat> bow_representation;

    //load dictionary
    Mat dictionary;
    FileStorage fs("D:/ee58j_final_project/dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();

    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

    auto detector = cv::SiftFeatureDetector::create();
    auto extractor = cv::SiftDescriptorExtractor::create();
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Dictionary is set with the vocabulary we created
    bowDE.setVocabulary(dictionary);

    cout <<" Obtaining BoW representation of test set and predicting..."<< endl;
    for (int i = 0; i < 10; ++i)
    {
        cout << files[i] << endl;

        std::string folder = "D:/ee58j_final_project/art_imgs_10/"+files[i]+"/*.jpg";
        std::vector<std::string> filenames;
        cv::glob(folder, filenames);
        random_shuffle(filenames.begin(), filenames.end());
        size_t N = filenames.size()*0.75 ;

        for (size_t c = N; c <filenames.size(); ++c)
        {
            //read the image
            Mat img=imread(filenames[c]);
            vector<KeyPoint> keypoints;
            detector->detect(img,keypoints);
            Mat bowDescriptor;

            //extract BoW  descriptor from given image
            bowDE.compute(img,keypoints,bowDescriptor);
            // create the BoW representation to feed SVM classifier
            bow_representation.push_back(bowDescriptor);
            // labels of the data
            test_labels->push_back(i);
           // float result=svm->predict(bowDescriptor);
            //cout << "Actual label:" << i << " Predicted label:" << result << endl;

        }

    }
    cout << "Obtaining BoW representation of test set done.."<< endl;
    return bow_representation;
}
// training the SVM
Ptr<SVM> SVMtrain(Mat trainMat, vector<int> trainLabels) {
     Mat testResponse;
    Ptr<SVM> svm = SVM::create();

    //svm->setC(10);
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
   // svm->setGamma(10);
    svm ->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);

    //svm->train(td);
    svm->trainAuto(td);

    cout << "SVM training complete..." << endl;
    svm->save("D:/ee58j_final_project/data.yml");
    cout << " SVM Model save complete..." << endl;

    return svm;
}

void ConvertVectortoMatrix(vector<Mat> &ipHOG, Mat & opMat)
{
    int descriptor_size = ipHOG[0].cols;
    for (int i = 0; i<ipHOG.size(); i++) {
        for (int j = 0; j<descriptor_size; j++) {
            Mat temp=ipHOG[i];
            opMat.at<float>(i, j) = temp.at<float>(0,j);
        }
    }
}

//evaluating the accuracy performance of the trained SVM
float SVMevaluate(Mat testResponse, float countt, vector<int> testLabels) {
    float accuracy;
    float class_acc;
    float class_count=0;
    int temp=0;
    for (int k = 0; k<20; k++)
    {
     int res = count(testLabels.begin(), testLabels.end(), k);
    for (int i = 0; i<res; i++)
    {
        if (testResponse.at<float>(temp+i, 0) == testLabels[temp+i]) {
            countt = countt + 1;
            class_count=class_count+1;
        }
    }
    temp=temp+res;
    class_acc=(class_count / res) * 100;
    class_count=0;
   cout << "Class "<<k<<" Accuracy:" <<class_acc << endl;

    }
        cout<<countt<<endl;
    accuracy = (countt / testResponse.rows) * 100;
    return accuracy;

}
//finds the best match of a random image and draws matches and saves into the current file
void find_best_match()
{


    auto detector = cv::SiftFeatureDetector::create();
    auto extractor = cv::SiftDescriptorExtractor::create();

    //random test image
    Mat img=imread("../576_project4/Scenes-5Places/Pl1/t1152873106.980797_x2.480106_y-0.302402_a-0.008646.jpeg");

    // the best match is found by experimenting
    Mat img2=imread("../576_project4/Scenes-5Places/Pl1/t1152873096.984173_x2.480106_y-0.302402_a-0.008646.jpeg");

    vector<KeyPoint> keypoints,keypoints2;
    detector->detect(img,keypoints);
    detector->detect(img2,keypoints2);

    Mat SIFTDescriptor,SIFTDescriptor2;
    //extract BoW  descriptor from given image
    extractor->compute(img,keypoints,SIFTDescriptor);
    extractor->compute(img2,keypoints2,SIFTDescriptor2);

    const float ratio_thresh = 0.7f;
    float dist;
    float min_dist=250000;
    int index=0;
//    for (size_t i = 0; i < SIFTDescriptors.size(); i++)
//    {
//        matcher->knnMatch( SIFTDescriptors[i], SIFTDescriptor, knn_matches, 2 );

//    for (size_t k = 0; k < knn_matches.size(); k++)
//    {
//        if (knn_matches[k][0].distance < ratio_thresh * knn_matches[k][1].distance)
//        {
//            dist=dist+knn_matches[k][0].distance;
//        }
//    }
//    if (dist < min_dist)
//    {
//        std::vector< std::vector<DMatch> > min_knn_matches=knn_matches;
//        min_dist=dist;
//        index=i;
//    }
//}

    //-- Filter matches using the Lowe's ratio test
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
      std::vector< std::vector<DMatch> > knn_matches;
      matcher->knnMatch( SIFTDescriptor, SIFTDescriptor2, knn_matches, 2 );
      //-- Filter matches using the Lowe's ratio test
      std::vector<DMatch> good_matches;
      for (size_t i = 0; i < knn_matches.size(); i++)
      {
          if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
          {
              good_matches.push_back(knn_matches[i][0]);
          }
      }
      //-- Draw matches
      Mat img_matches;
      drawMatches( img, keypoints, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                   Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
      //-- Show detected matches
      imshow("Good Matches", img_matches );
      waitKey(0);
      imwrite("../576_project4/matched_img.bmp",img_matches);
}
