// Source codes for EE_576 Project 4
// Mehmet Yiğit Avcı
// Bogazici University, 2022

// necessary declarations
#include <header.h>
#include "header.cpp"


int main(int argc, char *argv[])
{
   // bag_of_words_create_dictionary();

    // obtaining BoW representation of train set
    vector<Mat> train_bow= bow_representation_extract_train(&train_labels,&SIFTDescriptors,&SIFTKeypoints);
    cout << "bow set size: "<< train_bow.size() << endl;
    cout << "bow one image cols: "<< train_bow[0].cols << endl;
    cout << "bow one image rows: "<< train_bow[0].rows << endl;
    cout << "train labels size: "<< train_labels.size() << endl;

    // convert the train set BoW to matrix
    Mat trainM(train_bow.size(), train_bow[0].cols,CV_32FC1);
    ConvertVectortoMatrix(train_bow, trainM);
    cout << "train Matrix size: "<< trainM.size()<< endl;

    // training the SVM
    cout<<"Training the SVM..."<<endl;
    Ptr<SVM> svm= SVMtrain(trainM, train_labels);

    // evaluate the performance of the model
    cout<<"Evaluating the SVM for training set..."<<endl;

    // predict the results
    svm->predict(trainM,testR);
    accuracy=  SVMevaluate(testR, countt, train_labels);
    cout << "The accuracy of the model: " << accuracy << endl;

    // obtaining BoW representation of test set
    vector<Mat> test_bow= bow_representation_extract_test(&test_labels,svm);

    // convert the test set BoW to matrix
    Mat testMat(test_bow.size(), test_bow[0].cols, CV_32FC1);
    ConvertVectortoMatrix(test_bow, testMat);
    cout << "ttest size: "<< testMat.size() << endl;




    // evaluate the performance of the model
    cout<<"Evaluating the SVM for test set..."<<endl;
    // predict the results
    svm->predict(testMat,testR);
    accuracy=  SVMevaluate(testR, countt, test_labels);
    cout << "The accuracy of the model: " << accuracy << endl;

    //finds the best match for a random image and draws matches
  //  find_best_match();
    return 0;
}
