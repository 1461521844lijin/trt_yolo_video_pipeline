#include "xcodec.h"
#include <iostream>
using namespace std;
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}


	//////////////////////////////////////////
	/// 创建编码上下文
	/// @para codec_id 编码器ID号，对应ffmpeg
	/// @return 编码上下文 ,失败返回nullptr
AVCodecContext* XCodec::Create(int codec_id, bool isencode)
{
	//1 找到编码器
	const AVCodec* codec = nullptr;
	if (isencode){
		codec = avcodec_find_encoder((AVCodecID)codec_id);
	}else{
		codec = avcodec_find_decoder((AVCodecID)codec_id);
	}
	if (!codec)
	{
		cerr << "avcodec_find_encoder failed!" << codec_id << endl;
		return nullptr;
	}
	//创建上下文

	auto c = avcodec_alloc_context3(codec);
	if (!c)
	{
		cerr << "avcodec_alloc_context3 failed!" << codec_id << endl;
		return nullptr;
	}
	//设置参数默认值
	c->time_base = { 1,25 };
	c->pix_fmt = AV_PIX_FMT_YUV420P;
	c->thread_count = 8;
	if(isencode){
		c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; //全局参数
		c->codec_id = codec->id;
		c->width = 1920;
		c->height = 1080;
		c->gop_size = 25;
		c->max_b_frames = 0;
		c->pix_fmt = AV_PIX_FMT_YUV420P;	
	}
	return c;

}

AVCodecContext* XCodec::CreateHwEncode(std::string codec_name, int width, int height){

	const AVCodec* codec = nullptr;
	codec = avcodec_find_encoder_by_name(codec_name.c_str());
	if (!codec)
	{
		cerr << "avcodec_find_encoder failed!" << codec_name << endl;
		return nullptr;
	}
	//创建上下文
	auto context = avcodec_alloc_context3(codec);
	if (!context)
	{
		cerr << "avcodec_alloc_context3 failed!" << codec_name << endl;
		return nullptr;
	}
	//设置参数默认值
	context->time_base = { 1,25 };
	context->pix_fmt = AV_PIX_FMT_YUV420P;
	context->thread_count = 8;
	context->width = width;
	context->height = height;
	return context;

}


//////////////////////////////////////////
/// 设置对象的编码器上下文 上下文传递到对象中，资源由XEncode维护
/// 加锁 线程安全
/// @para c 编码器上下文 如果c_不为nullptr，则先清理资源
void XCodec::set_c(AVCodecContext* c)
{
	unique_lock<mutex>lock(mux_);
	if (c_)
	{
		avcodec_free_context(&c_);
	}
	this->c_ = c;
}


bool XCodec::SetOpt(const char* key, const char* val)
{
	unique_lock<mutex>lock(mux_);
	if (!c_)return false;
	auto re = av_opt_set(c_->priv_data, key, val, 0);
	if (re != 0)
	{
		cerr << "av_opt_set failed!";
		PrintErr(re);
	}
	return true;
}

bool XCodec::SetOpt(const char* key, int val)
{
	unique_lock<mutex>lock(mux_);
	if (!c_)return false;
	auto re = av_opt_set_int(c_->priv_data, key, val, 0);
	if (re != 0)
	{
		cerr << "av_opt_set failed!";
		PrintErr(re);
	}
	return true;
}

//////////////////////////////////////////////////////////////
/// 打开编码器 线程安全
bool XCodec::Open()
{
	unique_lock<mutex>lock(mux_);
	if (!c_)return false;
	auto re = avcodec_open2(c_, NULL, NULL);
	if (re != 0)
	{
		PrintErr(re);
		return false;
	}
	return true;
}


///////////////////////////////////////////////////////////////
//根据AVCodecContext 创建一个AVFrame，需要调用者释放av_frame_free
AVFrame* XCodec::CreateFrame()
{
	unique_lock<mutex>lock(mux_);
	if (!c_)return nullptr;
	auto frame = av_frame_alloc();
	frame->width = c_->width;
	frame->height = c_->height;
	frame->format = c_->pix_fmt;
	auto re = av_frame_get_buffer(frame, 0);
	if (re != 0)
	{
		av_frame_free(&frame);
		PrintErr(re);
		return nullptr;
	}
	return frame;
}
