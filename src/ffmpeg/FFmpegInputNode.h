#pragma once

#include <iostream>
#include <list>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "xdecode.h"
#include "xdemux.h"
#include "xencode.h"
#include "xmux.h"
#include "xscale.h"
#include "xtools.h"

#include "base/ProcessNode.hpp"
#include "base/TransferData.h"



extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

namespace FFmpeg {

using namespace std;

class FFmpegInputNode : public Base::Node {
public:
    typedef std::shared_ptr<FFmpegInputNode> ptr;

    /// <summary>
    /// 默认带参构造函数，内部通过三个参数来初始化解码器以及封装参数
    /// 如果输出图像不指定参数，将按照原始视频大小输出
    /// </summary>
    /// <param name="ffmpeg_name">类ID</param>
    /// <param name="stream_id">流ID</param>
    FFmpegInputNode(string ffmpeg_name, string stream_id);

    virtual ~FFmpegInputNode();

    bool Set_HW_decode(bool &flag);  // 设置是否使用硬件解码

    void reopen(const string &url);  // 改变视频流

    void Next(std::shared_ptr<Data::Decode_Data> data);  // 责任链执行函数

    string get_stream_id() const;
    void   set_strema_id(string &str);
    bool   get_is_decode() const;
    double get_current_skip_rate() const;
    int    get_input_width() const;
    int    get_input_height() const;

    void FrameTimeOut();
    void FrameInTime();

    virtual void worker();

private:
    string m_stream_id;                                  // 流ID
    double m_current_skip_rate = 0;                      // 丢帧间隔
    int    m_width_original = 0, m_height_original = 0;  // 原始图像宽，高
    int    m_width_convert = 0, m_height_convert = 0;    // 缩放后的宽、高
    int    m_bad_frame_count  = 0;                       // 近期丢帧数统计
    int    m_nice_frame_count = 0;                       // 正常帧计数

    double m_skip_len = 0.1;  // 丢帧间隔
    double m_max_skip_rate = 0.5;  // 最大丢帧率

    XDecode m_decode;
    XDemux  m_demux;
    XScale  m_scale_original;
};

shared_ptr<FFmpegInputNode>
create_ffmpeg(string ffmpeg_name, string stream_id);

}  // namespace FFmpeg