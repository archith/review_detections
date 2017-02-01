#include "DetectionReader.h"

using namespace std;

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    cout << "usage: readDetections video_filename detection_file" << endl;
    return -1;
  }

  string video_filename(argv[1]);
  string detection_file(argv[2]);

  DetectionReader detection_reader(video_filename, detection_file);
  cout << "Reading detection data" << endl;
  detection_reader.readDetections();
  cout << "Rendering detection outputs" << endl;
  detection_reader.renderDetections();


  return 0;
}