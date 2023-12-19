//
// Created by lijin on 2023/12/19.
//

#ifndef VIDEOPIPELINE_IVIDEOINFO_H
#define VIDEOPIPELINE_IVIDEOINFO_H

#include <string>

/*
 *  存放部分视频通用属性
 */
class IVideoInfo {
public:
    IVideoInfo()          = default;
    virtual ~IVideoInfo() = default;

public:
    void set_input_url(const std::string &input_url) {
        m_input_url = input_url;
    }
    std::string get_input_url() const {
        return m_input_url;
    }

    void set_output_url(const std::string &output_url) {
        m_output_url = output_url;
    }
    std::string get_output_url() const {
        return m_output_url;
    }

    void set_output_width(int output_width) {
        m_output_width = output_width;
    }
    int get_output_width() const {
        return m_output_width;
    }

    void set_output_height(int output_height) {
        m_output_height = output_height;
    }
    int get_output_height() const {
        return m_output_height;
    }

    void set_output_fps(int output_fps) {
        m_output_fps = output_fps;
    }
    int get_output_fps() const {
        return m_output_fps;
    }

    void set_output_bitrate(int output_bitrate) {
        m_output_bitrate = output_bitrate;
    }
    int get_output_bitrate() const {
        return m_output_bitrate;
    }

private:
    std::string m_input_url;   // rtsp url or video file path
    std::string m_output_url;  // rtmp url
    int         m_output_width   = 1920;
    int         m_output_height  = 1080;
    int         m_output_fps     = 25;
    int         m_output_bitrate = 1024 * 1024 * 2;
};

#endif  // VIDEOPIPELINE_IVIDEOINFO_H
