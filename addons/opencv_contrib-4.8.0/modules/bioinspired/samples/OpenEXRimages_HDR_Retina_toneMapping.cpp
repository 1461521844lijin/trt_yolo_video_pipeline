
//============================================================================
// Name        : OpenEXRimages_HDR_Retina_toneMapping.cpp
// Author      : Alexandre Benoit (benoit.alexandre.vision@gmail.com)
// Version     : 0.1
// Copyright   : Alexandre Benoit, LISTIC Lab, july 2011
// Description : HighDynamicRange retina tone mapping with the help of the Gipsa/Listic's retina in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstring>

#include "opencv2/bioinspired.hpp" // retina based algorithms
#include "opencv2/imgproc.hpp" // cvCvtcolor function
#include "opencv2/imgcodecs.hpp" // imread
#include "opencv2/highgui.hpp" // display

static void help(std::string errorMessage)
{
	std::cout<<"Program init error : "<<errorMessage<<std::endl;
	std::cout<<"\nProgram call procedure : ./OpenEXRimages_HDR_Retina_toneMapping [OpenEXR image to process]"<<std::endl;
	std::cout<<"\t[OpenEXR image to process] : the input HDR image to process, must be an OpenEXR format, see http://www.openexr.com/ to get some samples or create your own using camera bracketing and Photoshop or equivalent software for OpenEXR image synthesis"<<std::endl;
	std::cout<<"\nExamples:"<<std::endl;
	std::cout<<"\t-Image processing : ./OpenEXRimages_HDR_Retina_toneMapping memorial.exr"<<std::endl;
}

// simple procedure for 1D curve tracing
static void drawPlot(const cv::Mat curve, const std::string figureTitle, const int lowerLimit, const int upperLimit)
{
	//std::cout<<"curve size(h,w) = "<<curve.size().height<<", "<<curve.size().width<<std::endl;
	cv::Mat displayedCurveImage = cv::Mat::ones(200, curve.size().height, CV_8U);

	cv::Mat windowNormalizedCurve;
	normalize(curve, windowNormalizedCurve, 0, 200, cv::NORM_MINMAX, CV_32F);

	displayedCurveImage = cv::Scalar::all(255); // set a white background
	int binW = cvRound((double)displayedCurveImage.cols/curve.size().height);

	for( int i = 0; i < curve.size().height; i++ )
		rectangle( displayedCurveImage, cv::Point(i*binW, displayedCurveImage.rows),
				cv::Point((i+1)*binW, displayedCurveImage.rows - cvRound(windowNormalizedCurve.at<float>(i))),
				cv::Scalar::all(0), -1, 8, 0 );
	rectangle( displayedCurveImage, cv::Point(0, 0),
			cv::Point((lowerLimit)*binW, 200),
			cv::Scalar::all(128), -1, 8, 0 );
	rectangle( displayedCurveImage, cv::Point(displayedCurveImage.cols, 0),
			cv::Point((upperLimit)*binW, 200),
			cv::Scalar::all(128), -1, 8, 0 );

	cv::imshow(figureTitle, displayedCurveImage);
}
/*
 * objective : get the gray level map of the input image and rescale it to the range [0-255]
 */
static void rescaleGrayLevelMat(const cv::Mat &inputMat, cv::Mat &outputMat, const float histogramClippingLimit)
{

	// adjust output matrix wrt the input size but single channel
	std::cout<<"Input image rescaling with histogram edges cutting (in order to eliminate bad pixels created during the HDR image creation) :"<<std::endl;
	//std::cout<<"=> image size (h,w,channels) = "<<inputMat.size().height<<", "<<inputMat.size().width<<", "<<inputMat.channels()<<std::endl;
	//std::cout<<"=> pixel coding (nbchannel, bytes per channel) = "<<inputMat.elemSize()/inputMat.elemSize1()<<", "<<inputMat.elemSize1()<<std::endl;

	// rescale between 0-255, keeping floating point values
	cv::normalize(inputMat, outputMat, 0.0, 255.0, cv::NORM_MINMAX);

	// extract a 8bit image that will be used for histogram edge cut
	cv::Mat intGrayImage;
	if (inputMat.channels()==1)
	{
		outputMat.convertTo(intGrayImage, CV_8U);
	}else
	{
		cv::Mat rgbIntImg;
		outputMat.convertTo(rgbIntImg, CV_8UC3);
		cvtColor(rgbIntImg, intGrayImage, cv::COLOR_BGR2GRAY);
	}

	// get histogram density probability in order to cut values under above edges limits (here 5-95%)... useful for HDR pixel errors cancellation
	cv::Mat dst, hist;
	int histSize = 256;
	calcHist(&intGrayImage, 1, 0, cv::Mat(), hist, 1, &histSize, 0);
	cv::Mat normalizedHist;
	normalize(hist, normalizedHist, 1, 0, cv::NORM_L1, CV_32F); // normalize histogram so that its sum equals 1

	double min_val, max_val;
	minMaxLoc(normalizedHist, &min_val, &max_val);
	//std::cout<<"Hist max,min = "<<max_val<<", "<<min_val<<std::endl;

	// compute density probability
	cv::Mat denseProb=cv::Mat::zeros(normalizedHist.size(), CV_32F);
	denseProb.at<float>(0)=normalizedHist.at<float>(0);
	int histLowerLimit=0, histUpperLimit=0;
	for (int i=1;i<normalizedHist.size().height;++i)
	{
		denseProb.at<float>(i)=denseProb.at<float>(i-1)+normalizedHist.at<float>(i);
		//std::cout<<normalizedHist.at<float>(i)<<", "<<denseProb.at<float>(i)<<std::endl;
		if ( denseProb.at<float>(i)<histogramClippingLimit)
			histLowerLimit=i;
		if ( denseProb.at<float>(i)<1-histogramClippingLimit)
			histUpperLimit=i;
	}
	// deduce min and max admitted gray levels
	float minInputValue = (float)histLowerLimit/histSize*255;
	float maxInputValue = (float)histUpperLimit/histSize*255;

	std::cout<<"=> Histogram limits "
			<<"\n\t"<<histogramClippingLimit*100<<"% index = "<<histLowerLimit<<" => normalizedHist value = "<<denseProb.at<float>(histLowerLimit)<<" => input gray level = "<<minInputValue
			<<"\n\t"<<(1-histogramClippingLimit)*100<<"% index = "<<histUpperLimit<<" => normalizedHist value = "<<denseProb.at<float>(histUpperLimit)<<" => input gray level = "<<maxInputValue
			<<std::endl;
	//drawPlot(denseProb, "input histogram density probability", histLowerLimit, histUpperLimit);
	drawPlot(normalizedHist, "input histogram", histLowerLimit, histUpperLimit);

	// rescale image range [minInputValue-maxInputValue] to [0-255]
	outputMat-=minInputValue;
	outputMat*=255.0/(maxInputValue-minInputValue);
	// cut original histogram and back project to original image
	cv::threshold( outputMat, outputMat, 255.0, 255.0, 2 ); //THRESH_TRUNC, clips values above 255
	cv::threshold( outputMat, outputMat, 0.0, 0.0, 3 ); //THRESH_TOZERO, clips values under 0

}
// basic callback method for interface management
cv::Mat inputImage;
cv::Mat imageInputRescaled;
int histogramClippingValue;
static void callBack_rescaleGrayLevelMat(int, void*)
{
	std::cout<<"Histogram clipping value changed, current value = "<<histogramClippingValue<<std::endl;
	rescaleGrayLevelMat(inputImage, imageInputRescaled, (float)(histogramClippingValue/100.0));
	normalize(imageInputRescaled, imageInputRescaled, 0.0, 255.0, cv::NORM_MINMAX);
}

cv::Ptr<cv::bioinspired::Retina> retina;
int retinaHcellsGain;
int localAdaptation_photoreceptors, localAdaptation_Gcells;
static void callBack_updateRetinaParams(int, void*)
{
	retina->setupOPLandIPLParvoChannel(true, true, (float)(localAdaptation_photoreceptors/200.0), 0.5f, 0.43f, (float)retinaHcellsGain, 1.f, 7.f, (float)(localAdaptation_Gcells/200.0));
}

int colorSaturationFactor;
static void callback_saturateColors(int, void*)
{
	retina->setColorSaturation(true, (float)colorSaturationFactor);
}

int main(int argc, char* argv[])
{
	// welcome message
	std::cout<<"*********************************************************************************"<<std::endl;
	std::cout<<"* Retina demonstration for High Dynamic Range compression (tone-mapping) : demonstrates the use of a wrapper class of the Gipsa/Listic Labs retina model."<<std::endl;
	std::cout<<"* This retina model allows spatio-temporal image processing (applied on still images, video sequences)."<<std::endl;
	std::cout<<"* This demo focuses demonstration of the dynamic compression capabilities of the model"<<std::endl;
	std::cout<<"* => the main application is tone mapping of HDR images (i.e. see on a 8bit display a more than 8bits coded (up to 16bits) image with details in high and low luminance ranges"<<std::endl;
	std::cout<<"* The retina model still have the following properties:"<<std::endl;
	std::cout<<"* => It applies a spectral whithening (mid-frequency details enhancement)"<<std::endl;
	std::cout<<"* => high frequency spatio-temporal noise reduction"<<std::endl;
	std::cout<<"* => low frequency luminance to be reduced (luminance range compression)"<<std::endl;
	std::cout<<"* => local logarithmic luminance compression allows details to be enhanced in low light conditions\n"<<std::endl;
	std::cout<<"* for more information, reer to the following papers :"<<std::endl;
	std::cout<<"* Benoit A., Caplier A., Durette B., Herault, J., \"USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING\", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011"<<std::endl;
	std::cout<<"* Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891."<<std::endl;
	std::cout<<"* => reports comments/remarks at benoit.alexandre.vision@gmail.com"<<std::endl;
	std::cout<<"* => more informations and papers at : http://sites.google.com/site/benoitalexandrevision/"<<std::endl;
	std::cout<<"*********************************************************************************"<<std::endl;
	std::cout<<"** WARNING : this sample requires OpenCV to be configured with OpenEXR support **"<<std::endl;
	std::cout<<"*********************************************************************************"<<std::endl;
	std::cout<<"*** You can use free tools to generate OpenEXR images from images sets   :    ***"<<std::endl;
	std::cout<<"*** =>  1. take a set of photos from the same viewpoint using bracketing      ***"<<std::endl;
	std::cout<<"*** =>  2. generate an OpenEXR image with tools like qtpfsgui.sourceforge.net ***"<<std::endl;
	std::cout<<"*** =>  3. apply tone mapping with this program                               ***"<<std::endl;
	std::cout<<"*********************************************************************************"<<std::endl;

	// basic input arguments checking
	if (argc<2)
	{
		help("bad number of parameter");
		return -1;
	}

	bool useLogSampling = !strcmp(argv[argc-1], "log"); // check if user wants retina log sampling processing
	int chosenMethod=0;
	if (!strcmp(argv[argc-1], "fast"))
	{
		chosenMethod=1;
		std::cout<<"Using fast method (no spectral whithning), adaptation of Meylan&al 2008 method"<<std::endl;
	}

	std::string inputImageName=argv[1];

	//////////////////////////////////////////////////////////////////////////////
	// checking input media type (still image, video file, live video acquisition)
	std::cout<<"RetinaDemo: processing image "<<inputImageName<<std::endl;
	// image processing case
	// declare the retina input buffer... that will be fed differently in regard of the input media
	inputImage = cv::imread(inputImageName, -1); // load image in RGB mode
	std::cout<<"=> image size (h,w) = "<<inputImage.size().height<<", "<<inputImage.size().width<<std::endl;
	if (!inputImage.total())
	{
		help("could not load image, program end");
		return -1;
	}
	// rescale between 0 and 1
	normalize(inputImage, inputImage, 0.0, 1.0, cv::NORM_MINMAX);
	cv::Mat gammaTransformedImage;
	cv::pow(inputImage, 1./5, gammaTransformedImage); // apply gamma curve: img = img ** (1./5)
	imshow("EXR image original image, 16bits=>8bits linear rescaling ", inputImage);
	imshow("EXR image with basic processing : 16bits=>8bits with gamma correction", gammaTransformedImage);
	if (inputImage.empty())
	{
		help("Input image could not be loaded, aborting");
		return -1;
	}

	//////////////////////////////////////////////////////////////////////////////
	// Program start in a try/catch safety context (Retina may throw errors)
	try
	{
		/* create a retina instance with default parameters setup, uncomment the initialisation you wanna test
		 * -> if the last parameter is 'log', then activate log sampling (favour foveal vision and subsamples peripheral vision)
		 */
		if (useLogSampling)
		{
            retina = cv::bioinspired::Retina::create(inputImage.size(),true, cv::bioinspired::RETINA_COLOR_BAYER, true, 2.0, 10.0);
		}
		else// -> else allocate "classical" retina :
            retina = cv::bioinspired::Retina::create(inputImage.size());

		// create a fast retina tone mapper (Meyla&al algorithm)
		std::cout<<"Allocating fast tone mapper..."<<std::endl;
		//cv::Ptr<cv::RetinaFastToneMapping> fastToneMapper=createRetinaFastToneMapping(inputImage.size());
		std::cout<<"Fast tone mapper allocated"<<std::endl;

		// save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup"
		retina->write("RetinaDefaultParameters.xml");

		// desactivate Magnocellular pathway processing (motion information extraction) since it is not useful here
		retina->activateMovingContoursProcessing(false);

		// declare retina output buffers
		cv::Mat retinaOutput_parvo;

		/////////////////////////////////////////////
		// prepare displays and interactions
		histogramClippingValue=0; // default value... updated with interface slider
		//inputRescaleMat = inputImage;
		//outputRescaleMat = imageInputRescaled;
		cv::namedWindow("Processing configuration",1);
		cv::createTrackbar("histogram edges clipping limit", "Processing configuration",&histogramClippingValue,50,callBack_rescaleGrayLevelMat);

		colorSaturationFactor=3;
		cv::createTrackbar("Color saturation", "Processing configuration", &colorSaturationFactor,5,callback_saturateColors);

		retinaHcellsGain=40;
		cv::createTrackbar("Hcells gain", "Processing configuration",&retinaHcellsGain,100,callBack_updateRetinaParams);

		localAdaptation_photoreceptors=197;
		localAdaptation_Gcells=190;
		cv::createTrackbar("Ph sensitivity", "Processing configuration", &localAdaptation_photoreceptors,199,callBack_updateRetinaParams);
		cv::createTrackbar("Gcells sensitivity", "Processing configuration", &localAdaptation_Gcells,199,callBack_updateRetinaParams);


		/////////////////////////////////////////////
		// apply default parameters of user interaction variables
		rescaleGrayLevelMat(inputImage, imageInputRescaled, (float)histogramClippingValue/100);
		retina->setColorSaturation(true,(float)colorSaturationFactor);
		callBack_updateRetinaParams(1,NULL); // first call for default parameters setup

		// processing loop with stop condition
		bool continueProcessing=true;
		while(continueProcessing)
		{
			// run retina filter
			if (!chosenMethod)
			{
				retina->run(imageInputRescaled);
				// Retrieve and display retina output
				retina->getParvo(retinaOutput_parvo);
				cv::imshow("Retina input image (with cut edges histogram for basic pixels error avoidance)", imageInputRescaled/255.0);
				cv::imshow("Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping", retinaOutput_parvo);
				cv::imwrite("HDRinput.jpg",imageInputRescaled/255.0);
				cv::imwrite("RetinaToneMapping.jpg",retinaOutput_parvo);
			}
			else
			{
				// apply the simplified hdr tone mapping method
				cv::Mat fastToneMappingOutput;
				retina->applyFastToneMapping(imageInputRescaled, fastToneMappingOutput);
				cv::imshow("Retina fast tone mapping output : 16bit=>8bit image retina tonemapping", fastToneMappingOutput);
			}
			/*cv::Mat fastToneMappingOutput_specificObject;
             fastToneMapper->setup(3.f, 1.5f, 1.f);
             fastToneMapper->applyFastToneMapping(imageInputRescaled, fastToneMappingOutput_specificObject);
             cv::imshow("### Retina fast tone mapping output : 16bit=>8bit image retina tonemapping", fastToneMappingOutput_specificObject);
			 */
			cv::waitKey(10);
		}
	} catch(const cv::Exception& e)
	{
		std::cerr<<"Error using Retina : "<<e.what()<<std::endl;
	}

	// Program end message
	std::cout<<"Retina demo end"<<std::endl;

	return 0;
}
