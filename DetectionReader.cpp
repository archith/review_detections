#include "DetectionReader.h"


DetectionReader::DetectionReader(std::string video_filename, std::string detection_filename)
{
  
  _detect_fstream.open(detection_filename);
  if (!_detect_fstream.is_open())
  {
    std::cout << "Cannot open detection file!" << std::endl;
    throw std::runtime_error("Could not open detection file :" + detection_filename);
  }

  _video_cap.open(video_filename);
  if (!_video_cap.isOpened())
  {
    std::cout << "Cannot open video file!" << std::endl;
    throw std::runtime_error("Could not open video file :" + video_filename);
  }
}

DetectionReader::~DetectionReader()
{
  _video_cap.release();
  _detect_fstream.close();
}

void DetectionReader::readDetections()
{
  std::string line;
  std::getline(_detect_fstream, line); //get csv header out of the way
  while(std::getline(_detect_fstream, line))
  {
    std::vector<std::string> tokens;
    boost::split(tokens, line, boost::is_any_of(_csv_seperator));

    std::vector<double> line_det_bbox;
    int tok_counter = 0;
    for (std::vector<std::string>::iterator beg = tokens.begin(); beg!=tokens.end(); beg++)
    {
      if (tok_counter != _class_col_idx)
      {
        line_det_bbox.push_back(std::stod(*beg));
      }
      else
      {
        _det_class_data.push_back(*beg);
      }
      tok_counter++;
    }
    _det_bbox_data.push_back(line_det_bbox);
  }
  return;
}

void DetectionReader::renderDetections()
{
  cv::Mat frame;
  bool success;
  unsigned int frame_counter = 0;
  std::vector<std::vector<double>> frame_det_bboxes;
  std::vector<std::string> frame_det_classes;

  while(true)
  {
    success = _video_cap.read(frame);
    if (!success)
    {
      break;
    }
    // inscribe frame number
    cv::Scalar color = cv::Scalar(0, 255, 0); //Green?
    std::ostringstream frame_numstream;
    frame_numstream << frame_counter;
    std::string frame_numstring = frame_numstream.str();
    cv::Point frameNumPoint(0, 20);
    cv::putText(frame, frame_numstring, frameNumPoint, cv::FONT_HERSHEY_SIMPLEX, 
                1, color, 2);

    frame_det_bboxes = getFrameBboxes(frame_counter);
    frame_det_classes = getFrameClasses(frame_counter);
    drawAnnotations(frame, frame_det_bboxes, frame_det_classes, frame_counter);

    //resize for 
    double scale_ratio = 700/std::max(frame.rows, frame.cols);
    cv::resize(frame, frame, cv::Size(0,0), scale_ratio, scale_ratio);
    cv::imshow("Detection output", frame);
    cv::waitKey(30);
    frame_counter++;  
  }
  return;
}

std::vector<int> DetectionReader::getFrameMatchIndexes(int frame_number)
{
  std::vector<int> matched_frameidx;
  for (int idx = 0; idx < _det_bbox_data.size(); idx++)
  {
    int f = (int)_det_bbox_data[idx][_frame_col_idx];
    if (f == frame_number)
    {
      matched_frameidx.push_back(idx);
    } 
  }
  return matched_frameidx;
}

std::vector<std::vector<double>> DetectionReader::getFrameBboxes(int frame_number)
{
  std::vector<int> matched_frameIndexes;
  matched_frameIndexes = getFrameMatchIndexes(frame_number);

  std::vector<std::vector<double>> matched_frameBboxes;

  for (std::vector<int>::iterator iter = matched_frameIndexes.begin(); 
       iter !=  matched_frameIndexes.end(); iter++)
  {
    matched_frameBboxes.push_back(_det_bbox_data[*iter]);
  }

  return matched_frameBboxes;
}


std::vector<std::string> DetectionReader::getFrameClasses(int frame_number)
{
  std::vector<int> matched_frameIndexes;
  matched_frameIndexes = getFrameMatchIndexes(frame_number);

  std::vector<std::string> matched_frameClasses;

  for (std::vector<int>::iterator iter = matched_frameIndexes.begin(); 
       iter !=  matched_frameIndexes.end(); iter++)
  {
    matched_frameClasses.push_back(_det_class_data[*iter]);
  }

  return matched_frameClasses;
}
 
bool DetectionReader::validBbox(std::string bbox_class, std::vector<double> bbox_class_data)
{
  bool is_valid = true;
  is_valid = is_valid & (std::find(_classes_ointerest.begin(), 
                                   _classes_ointerest.end(), 
                                   bbox_class) != _classes_ointerest.end());
  is_valid = is_valid & (bbox_class_data[_confidence_col_idx] > _conf_threshold);

  return is_valid;
}

void DetectionReader::drawAnnotations(cv::Mat &frame, 
                                      std::vector<std::vector<double>>frame_bboxes, 
                                      std::vector<std::string> frame_classes,
                                      unsigned int frame_index)
{
  assert(frame_bboxes.size() == frame_classes.size());

  //iterate over and draw each box
  for (int i = 0; i < frame_bboxes.size(); i++)
  {
    // check if this box is valid in the context of the application
    bool box_validity = validBbox(frame_classes[i], frame_bboxes[i]);
    if (!box_validity)
    {
      continue;
    }

    int topX = (int)frame_bboxes[i][_topx_col_idx];
    int topY = (int)frame_bboxes[i][_topy_col_idx];
    int bottomX = (int)frame_bboxes[i][_bottomx_col_idx];
    int bottomY = (int)frame_bboxes[i][_bottomy_col_idx];
    double confidence = frame_bboxes[i][_confidence_col_idx];
    std::ostringstream string_stream; 
    string_stream << frame_classes[i] << " "  << confidence;
    std::string bbox_string = string_stream.str();
    
    cv::Scalar color = cv::Scalar(0, 255, 0); //Green?
    cv::Point topLeft(topX, topY);
    cv::Point bottomRight(bottomX, bottomY);
    // put boundingbox
    cv::rectangle(frame, topLeft, bottomRight, color, 2);
    
    //put boundingbox label
    cv::Point textPoint(topX, bottomY + 20);
    cv::putText(frame, bbox_string, textPoint, cv::FONT_HERSHEY_SIMPLEX, 
                1, color, 2);
    
                
  }
}
