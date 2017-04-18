#include <iostream>
#include "opencv2/opencv.hpp"
#include "emojis.h"

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;

vector<Rect> face_detection(Mat frame){
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );
    return faces;
}

int main() {

    vector<string> categories = {"Angry" , "Disgust" , "Fear" , "Happy"  , "Neutral" ,  "Sad" , "Surprise"};

    string face_cascade_name = "../haarcascade_frontalface_default.xml";

    string model_file = "../models/deploy.prototxt";

    string trained_file = "../models/EmotiW_VGG_S.caffemodel";

    string mean_file = "../models/mean.binaryproto";

    Emojis emojis(model_file, trained_file, mean_file);

    face_cascade.load( face_cascade_name );

    VideoCapture cap(0);

    Mat img;
    vector<Rect> faces;
    while(cap.read(img))
    {
        faces = face_detection(img);

        for(int i = 0; i < faces.size(); i++){
            rectangle(img, faces[i], cv::Scalar(0, 0, 255), 3);
        }

        for(int i = 0; i < faces.size(); i++) {
            int label;
            Mat img_resized = img(faces[i]);
            label = emojis.classify_emojis(img_resized);
            cv::putText(img, categories[label], cvPoint(faces[i].x + 3, faces[i].y + 13),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
        }
        imshow("Live", img);
        waitKey(1);
    }

    return 0;
}
