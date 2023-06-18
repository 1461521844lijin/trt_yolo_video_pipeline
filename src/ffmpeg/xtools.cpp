

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

AVPacket* XAVPacketList::Pop()
{
	unique_lock<mutex> lock(mux_);
	if (pkts_.empty())return nullptr;
	auto pkt = pkts_.front();
	pkts_.pop_front();
	return pkt;
}

void XAVPacketList::Push(AVPacket* pkt)
{
	//cout << "解码线程接收pkt\n" << flush;
	unique_lock<mutex> lock(mux_);
	//生成新的AVPacket 对象 引用计数+1;
	auto p = av_packet_alloc();
	av_packet_ref(p, pkt);//引用计数 减少数据复制，线程安全
	pkts_.push_back(p);

	//超出最大空间，清理数据，到关键帧位置
	if (pkts_.size() > max_packets_)
	{
		//处理第一帧
		if (pkts_.front()->flags & AV_PKT_FLAG_KEY)//关键帧
		{
			av_packet_free(&pkts_.front());//清理
			pkts_.pop_front();  //出队
			return;
		}
		//清理所有非关键帧之前的数据
		while (!pkts_.empty())
		{
			if (pkts_.front()->flags & AV_PKT_FLAG_KEY)//关键帧
			{
				return;
			}
			av_packet_free(&pkts_.front());//清理
			pkts_.pop_front();  //出队
		}
	}

}


AVFrame* XAVFrameList::Pop()
{
	unique_lock<mutex> lock(mux_);
	if (frames_.empty())return nullptr;
	auto frame = frames_.front();
	frames_.pop_front();
	return frame;
}

void XAVFrameList::Push(AVFrame* frame)
{
	unique_lock<mutex> lock(mux_);
	//生成新的AVPacket 对象 引用计数+1;
	//auto p = av_frame_alloc();
	//av_frame_ref(p, frame);//引用计数 减少数据复制，线程安全

	/*cout << "IN push frame_->width:" << frame->width << endl;
	cout << "frame->height:" << frame->height << endl;
	cout << "frame_->format:" << frame->format << endl;
	cout << "frame_->line:" << frame->linesize << endl;*/

	frames_.push_back(frame);

	//超出最大空间，清理数据，到关键帧位置
	if (frames_.size() > max_frames_)
	{

		//清空队列
		while (!frames_.empty())
		{

			av_frame_free(&frames_.front());//清理
			frames_.pop_front();  //出队
		}
	}

}
