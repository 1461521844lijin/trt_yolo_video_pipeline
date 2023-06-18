#include "xscale.h"
#include "xtools.h"

extern "C" {  // 指定函数是c语言函数，函数名不包含重载标注
// 引用ffmpeg头文件
#include "libavcodec/avcodec.h"
#include "libavdevice/avdevice.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/time.h"
#include "libswscale/swscale.h"
}
#include "FFmpegframe.h"
using namespace std;

SwsContext *XScale::InitScale(AVCodecParameters *para, int dstw, int dsth, int toformat) {

    SwsContext *sws =
        sws_getContext(para->width, para->height, (AVPixelFormat)para->format, dstw, dsth,
                       (AVPixelFormat)toformat, SWS_FAST_BILINEAR, NULL, NULL, NULL);

    sws_  = sws;
    para_ = para;
    return sws;
}


void XScale::InitScale(int srcw, int srch, int fromfromat, int dstw, int dsth, int toformat){
    auto sws = sws_getContext(srcw, srch, (AVPixelFormat)fromfromat, dstw, dsth,
                       (AVPixelFormat)toformat, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    sws_  = sws;
}



void XScale::Scale(AVFrame *frame, cv::Mat &image) {
    if (!frame) {
        cout << "scale frame is null" << endl;
        return ;
    }
    if (sws_) {
        cvLinesizes[0] = image.step1();

        int ret = sws_scale(sws_, frame->data, frame->linesize, 0, frame->height, &image.data,
                            cvLinesizes);

        return ;
    }

}

std::shared_ptr<AVFrame> XScale::Scale(cv::Mat& image){

    shared_ptr<AVFrame> yuv_frame = FFmpeg::alloc_av_frame();
    yuv_frame->width = image.cols;
    yuv_frame->height = image.rows;
    yuv_frame->format = AV_PIX_FMT_YUV420P;
    av_frame_get_buffer(yuv_frame.get(), 32);

    if (!yuv_frame) {
        cout << "scale frame is null" << endl;
        return nullptr;
    }

    if (sws_) {
        uint8_t* indata[AV_NUM_DATA_POINTERS] = { 0 };
		indata[0] = image.data;
		int insize[AV_NUM_DATA_POINTERS] = { 0 };
		//一行（宽）数据的字节数
		insize[0] = image.cols * image.elemSize();

        int ret = sws_scale(sws_, indata, insize, 0, image.rows, //源数据
			yuv_frame->data, yuv_frame->linesize);
        
        if (ret <= 0)
		{
			char buf[1024] = { 0 };
			av_strerror(ret, buf, sizeof(buf) - 1);
			printf(buf);
			return nullptr;
		}

        return yuv_frame;
    }


    return nullptr;

}
