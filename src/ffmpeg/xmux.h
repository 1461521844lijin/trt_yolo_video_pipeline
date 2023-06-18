#pragma once
#include "xformat.h"
//////////////////////////////////////
/// 媒体封装



class  XMux :public XFormat
{
public:
	//////////////////////////////////////////////////
	//// 打开封装
	static AVFormatContext* Open(const char* url, const char* format = NULL,
		AVCodecParameters* video_para = nullptr,
		AVCodecParameters* audio_para = nullptr
	);


	AVFormatContext* init_mux(const char* url, AVCodecContext* codec, const char* format = NULL);


	bool WriteHead();

	bool Write(AVPacket* pkt);

	bool WriteEnd();

	//音视频时间基础
	void set_src_video_time_base(AVRational* tb);
	void set_src_audio_time_base(AVRational* tb);

	~XMux();
private:
	AVRational* src_video_time_base_ = nullptr;
	AVRational* src_audio_time_base_ = nullptr;

	long long begin_video_pts_ = -1;//原视频开始时间
	long long begin_audio_pts_ = -1;//原音频开始时间
};

