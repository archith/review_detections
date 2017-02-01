#ifndef DetectionReader_h

#define DetectionReader_h
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include <cassert>
#include <vector>
#include <algorithm>

class DetectionReader
{
  private:
    std::ifstream _detect_fstream;
    cv::VideoCapture _video_cap;

    const int _frame_col_idx = 0;
    const int _topx_col_idx = 1;
    const int _topy_col_idx = 2;
    const int _bottomx_col_idx = 3;
    const int _bottomy_col_idx = 4;
    const int _confidence_col_idx = 5;
    const int _class_col_idx = 6;
    
    std::vector<std::vector<double>> _det_bbox_data;
    std::vector<std::string> _det_class_data;
    std::string _csv_seperator = std::string(",");
    std::vector<std::string> _classes_ointerest = std::vector<std::string>({"aeroplane", "bicycle", "bird", "boat",
    																		"bottle", "bus", "car", "cat", "chair",
    																	    "cow", "diningtable", "dog", "horse",
    																		"motorbike", "person", "pottedplant",
    																		"sheep", "sofa", "train", "tvmonitor"});
    double _conf_threshold = 0.3;


    std::vector<std::string> getFrameClasses(int frame_number);
    std::vector<std::vector<double>> getFrameBboxes(int frame_number);
    std::vector<int> getFrameMatchIndexes(int frame_number);
    void drawAnnotations(cv::Mat &frame, std::vector<std::vector<double>>frame_bboxes, 
                         std::vector<std::string> frame_classes, unsigned int frame_idx);
    bool validBbox(std::string bbox_class, std::vector<double> bbox_class_data);                         

  public:
    DetectionReader(std::string video_filename, std::string detection_filename);
    
    ~DetectionReader();

    void readDetections();
    
    void renderDetections();

};
#endif
