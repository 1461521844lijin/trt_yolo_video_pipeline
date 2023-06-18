

#include "xtools.h"
#include <sstream>
using namespace std;
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}



void PrintErr(int err)
{
	char buf[1024] = { 0 };
	av_strerror(err, buf, sizeof(buf) - 1);
	cerr << buf << endl;
}


void XFreeFrame(AVFrame** frame)
{
	if (!frame || !(*frame))return;
	av_frame_free(frame);
}
void MSleep(unsigned int ms)
{
	//auto beg = clock();
	//for (int i = 0; i < ms; i++)
	//{
	std::this_thread::sleep_for(std::chrono::microseconds(ms));
	//	usleep(1000);
	//	if ((clock() - beg) / (CLOCKS_PER_SEC / 1000) >= ms)
	//		break;
	//}
}
long long NowMs()
{
	return time(nullptr) * 1000;
}

////启动线程
//void XThread::Start()
//{
//	unique_lock<mutex> lock(m_);
//	static int i = 0;
//	i++;
//	index_ = i;
//	is_exit_ = false;
//	//启动线程
//	th_ = thread(&XThread::Main, this);
//	stringstream ss;
//	ss << "XThread::Start()" << index_;
//	LOGINFO(ss.str());
//}
//
////停止线程（设置退出标志，等待线程退出）
//void XThread::Stop()
//{
//	stringstream ss;
//	ss << "XThread::Stop() begin" << index_;
//	LOGINFO(ss.str());
//	is_exit_ = true;
//	if (th_.joinable()) //判断子线程是否可以等待
//		th_.join();     //等待子线程退出
//	ss.str("");
//	ss << "XThread::Stop() end" << index_;
//	LOGINFO(ss.str());
//}


//创建对象
XPara* XPara::Create()
{
	return new XPara();
}
XPara::~XPara()
{
	if (para)
	{
		avcodec_parameters_free(&para);
	}
	if (time_base)
	{
		delete time_base;
		time_base = nullptr;
	}
}

//私有是禁止创建栈中对象
XPara::XPara()
{
	para = avcodec_parameters_alloc();
	time_base = new AVRational();
}
