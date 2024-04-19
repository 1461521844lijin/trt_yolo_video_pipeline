#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::face;

static bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade)
{
    Mat gray;

    if (image.channels() > 1)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image.getMat().clone();

    equalizeHist(gray, gray);

    std::vector<Rect> faces_;
    face_cascade->detectMultiScale(gray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
    Mat(faces_).copyTo(faces);
    return true;
}

int main(int argc,char** argv){
    //Give the path to the directory containing all the files containing data
   CommandLineParser parser(argc, argv,
        "{ help h usage ?    |      | give the following arguments in following format }"
        "{ model_filename f  |      | (required) path to binary file storing the trained model which is to be loaded [example - /data/file.dat]}"
        "{ video v           |      | (required) path to video in which face landmarks have to be detected.[example - /data/video.avi] }"
        "{ face_cascade c    |      | Path to the face cascade xml file which you want to use as a detector}"
    );
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    string filename(parser.get<string>("model_filename"));
    if (filename.empty()){
        parser.printMessage();
        cerr << "The name  of  the model file to be loaded for detecting landmarks is not found" << endl;
        return -1;
    }
    string video(parser.get<string>("video"));
    if (video.empty()){
        parser.printMessage();
        cerr << "The name  of  the video file in which landmarks have to be detected is not found" << endl;
        return -1;
    }
    string cascade_name(parser.get<string>("face_cascade"));
    if (cascade_name.empty()){
        parser.printMessage();
        cerr << "The name of the cascade classifier to be loaded to detect faces is not found" << endl;
        return -1;
    }
    VideoCapture cap(video);
    if(!cap.isOpened()){
        cerr<<"Video cannot be loaded. Give correct path"<<endl;
        return -1;
    }
    //pass the face cascade xml file which you want to pass as a detector
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_name);
    FacemarkKazemi::Params params;
    Ptr<FacemarkKazemi> facemark = FacemarkKazemi::create(params);
    facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);
    facemark->loadModel(filename);
    cout<<"Loaded model"<<endl;
    //vector to store the faces detected in the image
    vector<Rect> faces;
    vector< vector<Point2f> > shapes;
    Mat img;
    while(1){
        faces.clear();
        shapes.clear();
        cap>>img;
        //Detect faces in the current image
        resize(img,img,Size(600,600), 0, 0, INTER_LINEAR_EXACT);
        facemark->getFaces(img,faces);
        if(faces.size()==0){
            cout<<"No faces found in this frame"<<endl;
        }
        else{
            for( size_t i = 0; i < faces.size(); i++ )
            {
                cv::rectangle(img,faces[i],Scalar( 255, 0, 0 ));
            }
            //vector to store the landmarks of all the faces in the image
            if(facemark->fit(img,faces,shapes))
            {
                for(unsigned long i=0;i<faces.size();i++){
                    for(unsigned long k=0;k<shapes[i].size();k++)
                        cv::circle(img,shapes[i][k],3,cv::Scalar(0,0,255),FILLED);
                }
            }
        }
        namedWindow("Detected_shape");
        imshow("Detected_shape",img);
        if(waitKey(1) >= 0) break;
    }
    return 0;
}