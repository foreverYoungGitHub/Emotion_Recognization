#define CPU_ONLY

#ifndef DEMO_EMOJIS_H
#define DEMO_EMOJIS_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace caffe;

class Emojis {
public:
    Emojis(const std::string model_file, const std::string trained_file, const std::string mean_file);
    ~Emojis();

    int classify_emojis(const cv::Mat& img);

private:
    void Preprocess(const cv::Mat &img);
    void SetMean(const std::string& mean_file);
    void Predict();
    void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels);

    std::shared_ptr<Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;

    cv::Mat img_;

    cv::Mat mean_;

    std::vector<float> labels_;
};


#endif //DEMO_EMOJIS_H
