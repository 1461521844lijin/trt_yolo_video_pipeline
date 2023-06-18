#include "xdecode.h"
#include <iostream>
using namespace std;
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}

bool XDecode::Send(const AVPacket *pkt)  // 发送解码
{
    unique_lock<mutex> lock(mux_);
    if (!c_)
        return false;
    auto re = avcodec_send_packet(c_, pkt);
    if (re != 0)
        return false;
    return true;
}

bool XDecode::Recv(AVFrame *frame)  // 获取解码
{
    unique_lock<mutex> lock(mux_);
    if (!c_)
        return false;
    auto f = frame;
    if (c_->hw_device_ctx)  // 硬件加速
    {
        f = av_frame_alloc();
    }
    auto re = avcodec_receive_frame(c_, f);
    if (re == 0) {
        if (c_->hw_device_ctx)  // GPU解码
        {
            // 显存转内存 GPU =》 CPU
            re = av_hwframe_transfer_data(frame, f, 0);
            av_frame_free(&f);
            if (re != 0) {
                PrintErr(re);
                return false;
            }
        }
        return true;
    } else {
        PrintErr(re);
    }
    if (c_->hw_device_ctx)
        av_frame_free(&f);
    return false;
}
bool XDecode::InitHW(int type) {
    unique_lock<mutex> lock(mux_);
    if (!c_)
        return false;
    ;
    AVBufferRef *ctx = nullptr;  // 硬件加速上下文
    auto         re  = av_hwdevice_ctx_create(&ctx, (AVHWDeviceType)type, NULL, NULL, 0);
    if (re != 0) {
        PrintErr(re);
        return false;
    }
    c_->hw_device_ctx = ctx;
    cout << "硬件加速：" << type << endl;
    return true;
}
std::vector<AVFrame *> XDecode::End()  // 获取缓存
{
    std::vector<AVFrame *> res;
    unique_lock<mutex>     lock(mux_);
    if (!c_)
        return res;

    /// 取出缓存数据
    int ret = avcodec_send_packet(c_, NULL);
    while (ret >= 0) {
        auto frame = av_frame_alloc();
        ret        = avcodec_receive_frame(c_, frame);
        if (ret < 0) {
            av_frame_free(&frame);
            break;
        }
        res.push_back(frame);
    }
    return res;
}
