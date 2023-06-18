

#include "xdemux.h"
#include <iostream>
#include <thread>
#include "xtools.h"
using namespace std;
extern "C" { //指定函数是c语言函数，函数名不包含重载标注
//引用ffmpeg头文件
#include <libavformat/avformat.h>
}


static int TimeoutCallback(void* para)
{
	auto xf = (XFormat*)para;
	if (xf->IsTimeout())return 1;//超时退出Read
	return 0; //正常阻塞
}

void PrintErr(int err);
#define BERR(err) if(err!= 0){PrintErr(err);return 0;}
AVFormatContext* XDemux::Open(const char* url)
{
	AVFormatContext* c = avformat_alloc_context();

	this->time_out_ms_ = 1000*10;
	last_time_ = NowMs();
	AVIOInterruptCB cb = { TimeoutCallback ,this };
	c->interrupt_callback = cb;

	AVDictionary* opts = nullptr;
	av_dict_set(&opts, "rtsp_transport", "tcp", 0); //传输媒体流为tcp协议，默认udp
	av_dict_set(&opts, "stimeout", "3000000", 0);  //连接超时3秒
	// av_dict_set(&opts, "rtbufsize", "100000", 0);

	//打开封装上下文
	auto re = avformat_open_input(&c, url, nullptr, &opts);
	if (opts)
		av_dict_free(&opts);
	BERR(re);
	//获取媒体信息
	re = avformat_find_stream_info(c, nullptr);
	BERR(re);
	//打印输入封装信息
	av_dump_format(c, 0, url, 0);

	return c;
}

int XDemux::Read(AVPacket* pkt)
{
	unique_lock<mutex> lock(mux_);
	if (!c_)return false;
	auto re = av_read_frame(c_, pkt);
	if(re != 0){
		PrintErr(re);
	}else{
		//计时 用于超时判断
		last_time_ = NowMs();
	}
	return re;
}

bool XDemux::Seek(long long pts, int stream_index)
{
	unique_lock<mutex> lock(mux_);
	if (!c_)return false;
	auto re = av_seek_frame(c_, stream_index, pts,
		AVSEEK_FLAG_FRAME | AVSEEK_FLAG_BACKWARD);
	BERR(re);
	return true;
}
