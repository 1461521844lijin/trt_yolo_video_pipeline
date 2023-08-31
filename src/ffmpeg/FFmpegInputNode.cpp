#include "FFmpegInputNode.h"
#include <chrono>

#include "FFmpegframe.h"

namespace FFmpeg {

    FFmpegInputNode::~FFmpegInputNode() {
        m_run = false;
    }


    FFmpegInputNode::FFmpegInputNode(string ffmpeg_name, string stream_id)
            : Node(ffmpeg_name),
              m_stream_id(stream_id) {
        // 打开解封装
        auto ic = m_demux.Open(stream_id.c_str());
        if (!ic) {
            throw std::runtime_error("open url failed");
        }
        // m_demux.set_time_out_ms(10000);  //设置超时时间10s
        // 设置解码上下文
        m_demux.set_c(ic);

        // 复制参数
        auto para = m_demux.CopyVideoPara();

        // 设置解码上下文
        auto c = m_decode.Create(m_demux.video_codec_id(), false);

        // 打开本地文件时，需要将视频容器中的参数copy到解码器中
        // 如果打开在线摄像头则不需要这句
        avcodec_parameters_to_context(c, ic->streams[m_demux->video_index()]->codecpar);

        m_decode.set_c(c);

#ifdef USE_CUDA_HW
        // 设置硬件cuda解码
        if (m_decode.InitHW(AV_HWDEVICE_TYPE_CUDA)) {
             para->para->format = AV_PIX_FMT_NV12;
        }
#endif
        // 打开解码器
        m_decode.Open();

        m_width_original = para->para->width;
        m_height_original = para->para->height;

        // 初始化格式转换上下文, 按照原始输入尺寸转换
        m_scale_original.InitScale(para->para, para->para->width, para->para->height,
                                   AV_PIX_FMT_BGR24);
    }

    void FFmpegInputNode::worker() {
        auto pkt = alloc_av_packet();
        auto frame = alloc_av_frame();
        double acc_skip_rate = 0;
        while (m_run) {
            auto re = m_demux.Read(pkt.get());
            if (re == 0) {
                if (pkt->stream_index != m_demux.video_index()) {
                    continue;  // 忽略音频包
                }
                m_decode.Send(pkt.get());
                av_packet_unref(pkt.get());
                if (!m_decode.Recv(frame.get())) {
                    continue;  // 解码器内部有缓冲，前几帧接收不到
                }
                std::shared_ptr<Data::Decode_Data> decode_data = std::make_shared<Data::Decode_Data>(
                        m_width_original, m_height_original);
                m_scale_original.Scale(frame.get(), decode_data->original_image);
                acc_skip_rate += m_current_skip_rate;
                if (acc_skip_rate < 1.0) {
                    Next(decode_data);
                } else {
                    acc_skip_rate -= 1.0;
                }
                // 打开路数较多时可以适当减少休眠时间
                this_thread::sleep_for(chrono::milliseconds(30));

            } else {
                if (!m_demux.is_connected()) {
                    char buf[1024] = {0};
                    av_strerror(re, buf, sizeof(buf) - 1);
                    // LOG(WARNING) << buf << " ____ " << get_stream_id() << "  reopen" << std::endl;
                    reopen(m_stream_id);
                    this_thread::sleep_for(chrono::milliseconds(10));
                    continue;
                } else {
                    if (re == AVERROR_EOF && m_stream_id.substr(0, 4) != "rtsp") {
                        // LOG(WARNING) << "视频解码完毕，任务结束";
                        break;
                    }
                }
            }
        }
    }

    void FFmpegInputNode::Next(std::shared_ptr<Data::Decode_Data> data) {
        for (auto next: m_output_buffers) {
            auto re = next.second->Push(data);
            if (re) {
                FrameInTime();
            } else {
                FrameTimeOut();
            }
        }
    }

    void FFmpegInputNode::reopen(const string &url) {
        auto ic = m_demux.Open(url.c_str());
        if (!ic) {
            throw std::runtime_error("open url failed");
            return;
        } else {

        }
        m_demux.set_c(ic);
    }

    string FFmpegInputNode::get_stream_id() const {
        return m_stream_id;
    }

    void FFmpegInputNode::set_strema_id(string &str) {
        m_stream_id = str;
    }

    bool FFmpegInputNode::get_is_decode() const {
        return m_run;
    }

    double FFmpegInputNode::get_current_skip_rate() const {
        return m_current_skip_rate;
    }

    int FFmpegInputNode::get_input_width() const {
        return m_width_original;
    }

    int FFmpegInputNode::get_input_height() const {
        return m_height_original;
    }

    void FFmpegInputNode::FrameTimeOut() {
        m_nice_frame_count = 0;
        m_bad_frame_count++;
        if (m_bad_frame_count > 5) {
            m_bad_frame_count = 0;
            m_current_skip_rate += m_skip_len;
            // 如果达到了设定的最大值则不再允许增加
            if (m_current_skip_rate > m_max_skip_rate) {
                m_current_skip_rate = m_max_skip_rate;
            }
        }
    }

    void FFmpegInputNode::FrameInTime() {
        m_bad_frame_count = 0;
        m_nice_frame_count++;
        if (m_nice_frame_count > 500) {
            m_nice_frame_count = 0;
            m_current_skip_rate -= m_skip_len;
            if (m_current_skip_rate <= 0.0) {
                m_current_skip_rate = 0.0;
            }
        }
    }

    FFmpegInputNode::ptr create_ffmpeg(string ffmpeg_name, string stream_id) {
        shared_ptr<FFmpegInputNode> instance(new FFmpegInputNode(ffmpeg_name, stream_id));
        return instance;
    }

}  // namespace FFmpeg
