#pragma once
#include "xtools.h"
/// <summary>
/// 封装和解封装基类
/// </summary>
#include <mutex>




struct AVFormatContext;
struct AVCodecParameters;
struct AVPacket;
struct AVCodecContext;
struct XRational
{
	int num; ///< Numerator
	int den; ///< Denominator
};
class  XFormat
{
public:
	/// <summary>
	/// 复制参数 线程安全
	/// </summary>
	/// <param name="stream_index">对应c_->streams 下标</param>
	/// <param name="dst">输出参数</param>
	/// <returns>是否成功</returns>
	bool CopyPara(int stream_index, AVCodecParameters* dst);
	bool CopyPara(int stream_index, AVCodecContext* dts);

	//返回智能指针 复制视频参数
	std::shared_ptr<XPara> CopyVideoPara();
	std::shared_ptr<XPara> CopyAudioPara();
	/// <summary>
	/// 设置上下文，并且清理上次的设置的值，如果传递NULL，相当于关闭上下文3
	/// 线程安全
	/// </summary>
	/// <param name="c"></param>
	void set_c(AVFormatContext* c);
	int audio_index() { return audio_index_; }
	int video_index() { return video_index_; }
	XRational video_time_base() { return video_time_base_; }
	XRational audio_time_base() { return audio_time_base_; }

	//根据timebase换算时间
	bool RescaleTime(AVPacket* pkt, long long offset_pts, XRational time_base);
	bool RescaleTime(AVPacket* pkt, long long offset_pts, AVRational* time_base);
	//把pts dts duration 值转为毫秒
	long long RescaleToMs(long long pts, int index);

	int video_codec_id() { return video_codec_id_; }

	//判断是否超时
	bool IsTimeout()
	{
		auto now = NowMs();
		if (now - last_time_ > time_out_ms_) //超时
		{
			last_time_ = NowMs();
			std::cout << "XFormat::IsTimeout" << std::endl;
			is_connected_ = false;
			return true;
		}

		return false;
	}

	//设定超时时间
	void set_time_out_ms(int ms);

	bool is_connected() { return is_connected_; }
protected:
	int time_out_ms_ = 0;           //超时时间 毫秒
	long long last_time_ = 0;       //上次接收到数据的时间
	bool is_connected_ = false;     //是否连接成功
	AVFormatContext* c_ = nullptr;  //封装解封装上下文
	std::mutex mux_;                //c_ 资源互斥
	int video_index_ = 0;           //video和audio在stream中索引
	int audio_index_ = 1;
	XRational video_time_base_ = { 1,25 };
	XRational audio_time_base_ = { 1,9000 };
	int video_codec_id_ = 0;       //编码器ID
};

