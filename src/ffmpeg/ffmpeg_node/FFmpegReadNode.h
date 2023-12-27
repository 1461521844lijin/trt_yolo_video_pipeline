//
// Created by lijin on 2023/12/20.
//

#ifndef VIDEOPIPELINE_FFMPEGREADNODE_H
#define VIDEOPIPELINE_FFMPEGREADNODE_H

#include "graph/core/node/ProcessNode.h"

#include "ffmpeg/core/Decoder.h"
#include "ffmpeg/core/Demuxer.h"
#include "ffmpeg/core/Scaler.h"

namespace Node {
class FFmpegReadNode : public GraphCore::Node {
public:
    typedef std::shared_ptr<FFmpegReadNode> ptr;

    /**
     *
     * @param name  节点名称
     * @param open_source  读取的文件路径或流地址
     * @param use_hw  是否使用硬件加速
     * @param cycle     如果是文件，是否循环读取
     */
    explicit FFmpegReadNode(const std::string &name,
                            std::string        open_source,
                            bool               use_hw = false,
                            bool               cycle  = false);

    ~FFmpegReadNode() override;

    static FFmpegReadNode::ptr CreateShared(const std::string &name,
                                            std::string        open_source,
                                            bool               use_hw = false,
                                            bool               cycle  = false);

    bool Init() override;

public:
    // 接收宽、高、帧率、码率数据返回
    std::tuple<int, int, int, int64_t> get_video_info() const;

private:
    void worker() override;

private:
    std::string m_open_source;     // 读取的文件路径或流地址
    bool        m_cycle  = false;  // 如果是文件，是否循环读取
    bool        m_use_hw = false;  // 是否使用硬件加速

    // 读取的视频信息
    int     m_width   = 0;
    int     m_height  = 0;
    int     m_fps     = 0;
    int64_t m_bitrate = 0;

private:
    std::shared_ptr<FFmpeg::Scaler>  m_scaler;   // 视频缩放、格式转换
    std::shared_ptr<FFmpeg::Demuxer> m_demux;    // 解封装
    std::shared_ptr<FFmpeg::Decoder> m_decoder;  // 解码
};
}  // namespace Node
#endif  // VIDEOPIPELINE_FFMPEGREADNODE_H
