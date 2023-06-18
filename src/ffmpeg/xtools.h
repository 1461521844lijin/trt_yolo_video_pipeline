#pragma once

#include <thread>
#include <iostream>
#include <mutex>
#include <list>
#include<unistd.h>
#include <chrono>



struct AVPacket;
struct AVCodecParameters;
struct AVRational;
struct AVFrame;
struct AVCodecContext;


//日志级别 DEBUG INFO ERROR FATAL
enum XLogLevel
{
	XLOG_TYPE_DEBUG,
	XLOG_TYPE_INFO,
	XLOG_TPYE_ERROR,
	XLOG_TYPE_FATAL
};
#define LOG_MIN_LEVEL XLOG_TYPE_DEBUG
#define XLOG(s,level) \
    if(level>=LOG_MIN_LEVEL) \
    std::cout<<level<<":"<<__FILE__<<":"<<__LINE__<<":\n"\
    <<s<<std::endl;
#define LOGDEBUG(s) XLOG(s,XLOG_TYPE_DEBUG)
#define LOGINFO(s) XLOG(s,XLOG_TYPE_INFO)
#define LOGERROR(s) XLOG(s,XLOG_TPYE_ERROR)
#define LOGFATAL(s) XLOG(s,XLOG_TYPE_FATAL)

void MSleep(unsigned int ms);

//获取当前时间戳 毫秒
 long long NowMs();

 void XFreeFrame(AVFrame** frame);

 void PrintErr(int err);




//音视频参数
class  XPara
{
public:
	AVCodecParameters* para = nullptr;  //音视频参数
	AVRational* time_base = nullptr;    //时间基数

	//创建对象
	static XPara* Create();
	~XPara();
private:
	//私有是禁止创建栈中对象
	XPara();
};
