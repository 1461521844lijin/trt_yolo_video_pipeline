//
// Created by lijin on 2023/12/20.
//

#ifndef VIDEOPIPELINE_FFMPEGOUTPUTNODE_H
#define VIDEOPIPELINE_FFMPEGOUTPUTNODE_H

#include "ffmpeg/core/Encoder.h"
#include "ffmpeg/core/Enmuxer.h"
#include "ffmpeg/core/Scaler.h"
#include "graph/core/node/ProcessNode.h"
namespace Node {

class FFmpegOutputNode : public GraphCore::Node {
public:
    typedef std::shared_ptr<FFmpegOutputNode> ptr;

    /**
     * @brief 编码节点
     * @param name          节点名称
     * @param open_source   读取的文件路径或流地址
     * @param from_width    输入宽度
     * @param from_height   输入高度
     * @param from_format   输入格式
     * @param to_width      输出宽度
     * @param to_height     输出高度
     * @param to_format     输出格式
     * @param fps           帧率
     * @param m_bitrate     码率
     * @param m_use_hw      是否使用硬编码
     */
    FFmpegOutputNode(std::string name,
                     std::string open_source,
                     int         from_width,
                     int         from_height,
                     int         from_format,
                     int         to_width,
                     int         to_height,
                     int         to_format,
                     int         fps     = 25,
                     int         bitrate = 1024 * 1024 * 2,
                     bool        use_hw  = false);

    virtual ~FFmpegOutputNode();

    static FFmpegOutputNode::ptr CreateShared(std::string        name,
                                              const std::string &open_source,
                                              int                from_width,
                                              int                from_height,
                                              int                from_format,
                                              int                to_width,
                                              int                to_height,
                                              int                to_format,
                                              int                fps     = 25,
                                              int                bitrate = 1024 * 1024 * 2,
                                              bool               use_hw  = false);

public:
private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;

protected:
    int  m_from_width  = 0;
    int  m_from_height = 0;
    int  m_from_format = 0;
    int  m_to_width    = 0;
    int  m_to_height   = 0;
    int  m_to_format   = 0;
    int  m_fps         = 25;
    int  m_bitrate     = 1024 * 1024 * 2;
    bool m_use_hw      = false;

    // 编码格式
    AVCodecID m_codec_id = AV_CODEC_ID_H264;

protected:
    std::string                      m_open_source;  // 读取的文件路径或流地址
    std::shared_ptr<FFmpeg::Enmuxer> m_enmux;
    std::shared_ptr<FFmpeg::Encoder> m_encoder;
    std::shared_ptr<FFmpeg::Scaler>  m_scaler;

    av_frame m_yuv_frame = alloc_av_frame();
    int      pts         = 0;
};

}  // namespace Node

#endif  // VIDEOPIPELINE_FFMPEGOUTPUTNODE_H
