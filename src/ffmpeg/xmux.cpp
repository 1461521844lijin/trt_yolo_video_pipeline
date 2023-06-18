
#include "xmux.h"

#include <iostream>
#include <thread>
using namespace std;
extern "C" { //指定函数是c语言函数，函数名不包含重载标注
//引用ffmpeg头文件
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}



void PrintErr(int err);

#define BERR(err) if(err!= 0){PrintErr(err);return 0;}
void XMux::set_src_video_time_base(AVRational* tb)
{
	if (!tb)return;
	unique_lock<mutex> lock(mux_);
	if (!src_video_time_base_)
		src_video_time_base_ = new AVRational();
	*src_video_time_base_ = *tb;
}
void XMux::set_src_audio_time_base(AVRational* tb)
{
	if (!tb)return;
	unique_lock<mutex> lock(mux_);
	if (!src_audio_time_base_)
		src_audio_time_base_ = new AVRational();
	*src_audio_time_base_ = *tb;
}
XMux::~XMux()
{
	unique_lock<mutex> lock(mux_);
	if (src_video_time_base_)
		delete src_video_time_base_;
	src_video_time_base_ = nullptr;
	if (src_audio_time_base_)
		delete src_audio_time_base_;
	src_audio_time_base_ = nullptr;
}

//////////////////////////////////////////////////
//// 打开封装
AVFormatContext* XMux::Open(const char* url, const char* format,
	AVCodecParameters* video_para,
	AVCodecParameters* audio_para)
{
	AVFormatContext* c = nullptr;
	//创建上下文
	auto re = avformat_alloc_output_context2(&c, NULL, format, url);
	BERR(re);


	//添加视频音频流
	if (video_para)
	{
		auto vs = avformat_new_stream(c, NULL);   //视频流
		avcodec_parameters_copy(vs->codecpar, video_para);
	}
	if (audio_para)
	{
		auto as = avformat_new_stream(c, NULL);   //音频流
		avcodec_parameters_copy(as->codecpar, audio_para);
	}

	//打开IO
	re = avio_open(&c->pb, url, AVIO_FLAG_WRITE);
	BERR(re);
	av_dump_format(c, 0, url, 1);
	return c;
}


AVFormatContext* XMux::init_mux(const char* url, AVCodecContext* codec, const char* format) {


	AVFormatContext* c = nullptr;
	//创建上下文
	auto re = avformat_alloc_output_context2(&c, NULL, format, url);

	AVStream* vs = avformat_new_stream(c, NULL);

	vs->codecpar->codec_tag = 0;
	//从编码器复制参数
	avcodec_parameters_from_context(vs->codecpar, codec);

	re = avio_open(&c->pb, url, AVIO_FLAG_WRITE);

	av_dump_format(c, 0, url, 1);
	c_ = c;
	return c;

}


bool XMux::Write(AVPacket* pkt)
{
	if (!pkt)return false;
	unique_lock<mutex> lock(mux_);
	if (!c_)return false;
	//没读取到pts 重构考虑通过duration 计算
	if (pkt->pts == AV_NOPTS_VALUE)
	{
		pkt->pts = 0;
		pkt->dts = 0;
	}
	if (pkt->stream_index == video_index_)
	{
		if (begin_video_pts_ < 0)
			begin_video_pts_ = pkt->pts;
		lock.unlock();
		RescaleTime(pkt, begin_video_pts_, src_video_time_base_);
		lock.lock();

	}
	else if (pkt->stream_index == audio_index_)
	{
		if (begin_audio_pts_ < 0)
			begin_audio_pts_ = pkt->pts;
		lock.unlock();
		RescaleTime(pkt, begin_audio_pts_, src_audio_time_base_);
		lock.lock();
	}

	//cout << pkt->pts << " " << flush;
	//写入一帧数据，内部缓冲排序dts，通过pkt=null 可以写入缓冲
	auto re = av_interleaved_write_frame(c_, pkt);
	BERR(re);
	return true;
}

bool XMux::WriteEnd()
{
	unique_lock<mutex> lock(mux_);
	if (!c_)return false;
	int re = 0;
	//auto re = av_interleaved_write_frame(c_, nullptr);//写入排序缓冲
	//BERR(re);
	re = av_write_trailer(c_);
	BERR(re);
	return true;
}
bool XMux::WriteHead()
{
	unique_lock<mutex> lock(mux_);
	if (!c_)return false;
	//会改变timebase
	auto re = avformat_write_header(c_, nullptr);
	BERR(re);

	//打印输出上下文
	av_dump_format(c_, 0, c_->url, 1);
	this->begin_audio_pts_ = -1;
	this->begin_video_pts_ = -1;


	return true;
}
