

#include "xencode.h"
#include <iostream>
using namespace std;
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}


//////////////////////////////////////////////////////////////
/// 编码数据 线程安全 每次新创建AVPacket
/// @para frame 空间由用户维护
/// @return 失败范围nullptr 返回的AVPacket用户需要通过av_packet_free 清理
AVPacket* XEncode::Encode(AVFrame* frame)
{
	if (!frame)return nullptr;

	unique_lock<mutex>lock(mux_);
	if (!c_)return nullptr;
	av_frame_make_writable((AVFrame*)frame);
	frame->pts = pts_++;
	//发送到编码线程
	auto re = avcodec_send_frame(c_, frame);
	if (re != 0)return nullptr;
	auto pkt = av_packet_alloc();
	//接收编码线程数据
	re = avcodec_receive_packet(c_, pkt);
	if (re == 0)
	{
		return pkt;
	}
	av_packet_free(&pkt);
	if (re == AVERROR(EAGAIN) || re == AVERROR_EOF)
	{
		return nullptr;
	}
	if (re < 0)
	{
		PrintErr(re);
	}
	return nullptr;

}

//////////////////////////////////////////////////////////////
//返回所有编码缓存中AVPacket
std::vector<AVPacket*> XEncode::End()
{
	std::vector<AVPacket*> res;
	unique_lock<mutex>lock(mux_);
	if (!c_)return res;
	auto re = avcodec_send_frame(c_, NULL); //发送NULL 获取缓冲
	if (re != 0)return res;
	while (re >= 0)
	{
		auto pkt = av_packet_alloc();
		re = avcodec_receive_packet(c_, pkt);
		if (re != 0)
		{
			av_packet_free(&pkt);
			break;
		}
		res.push_back(pkt);
	}
	return res;
}
