#include "emojis.h"

Emojis::Emojis(const std::string model_file, const std::string trained_file, const std::string mean_file){
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    std::shared_ptr<Net<float>> net;

    cv::Size input_geometry;
    int num_channel;

    net.reset(new Net<float>(model_file, TEST));
    net->CopyTrainedLayersFrom(trained_file);

    Blob<float>* input_layer = net->input_blobs()[0];
    num_channel = input_layer->channels();
    input_geometry = cv::Size(input_layer->width(), input_layer->height());

    net_ = net;
    input_geometry_ = input_geometry;
    num_channels_ = num_channel;
    SetMean(mean_file);
}

Emojis::~Emojis(){

}

int Emojis::classify_emojis(const cv::Mat& img){

    Preprocess(img);

    Predict();

    int label;
    float label_value = 0.0f;
    for(int i = 0; i < labels_.size(); i ++){
        if (labels_[i] > label_value){
            label = i;
            label_value = labels_[i];
        }
    }

    return label;
}

void Emojis::Preprocess(const cv::Mat &img){
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    img_ = sample_normalized;
}

/* Load the mean file in binaryproto format. */
void Emojis::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

void Emojis::Predict() {
    std::shared_ptr<Net<float>> net = net_;
    cv::Mat img = img_;

    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(img, &input_channels);
    net->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* label = net->output_blobs()[0];

    const float* label_begin = label->cpu_data();
    const float* label_end = label_begin + label->channels();
    labels_ = std::vector<float>(label_begin, label_end);
}

void Emojis::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels){
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int j = 0; j < input_layer->channels(); ++j)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(img, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

