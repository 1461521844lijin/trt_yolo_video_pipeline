#pragma once
#include "xformat.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


struct SwsContext;

class XScale
{
public:
	//输入：解码器参数、目标像素宽与高、目标像素格式
	SwsContext* InitScale(AVCodecParameters* para, int dstw, int dsth, int toformat);

	void InitScale(int srcw, int srch, int fromfromat, int dstw, int dsth, int toformat);


	void Scale(AVFrame* frame, cv::Mat& image);

	std::shared_ptr<AVFrame> Scale(cv::Mat& image);

private:

	SwsContext* sws_;
	AVCodecParameters* para_;

	AVFrame* bgrframe_;
	uint8_t* buffer_;
	size_t numBytes_;
	size_t bufsize_;


	int cvLinesizes[1];

};
