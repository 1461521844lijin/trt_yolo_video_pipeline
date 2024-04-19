#ifndef __HTTPSERVICE__H__
#define __HTTPSERVICE__H__

#include <string>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include "json.hpp"
#include "curl/curl.h"
#include "logger.hpp"

class HttpService{
    public:
    HttpService();
    ~HttpService();
    static bool is_base64(const char c);
    std::string base64_encode(const unsigned char * bytes_to_encode, unsigned int in_len);
    std::string base64_decode(std::string const & encoded_string);
    std::string b64_encode(cv::Mat);

    // post
	std::string requesthttp(std::string sendstr, std::string url);

    private:
    std::vector<uchar> m_vec_img;
    std::vector<int> m_vec_compresion;
    const std::string m_base64_chars = 
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

    // httpservice
    std::string m_keyframe_url;

};


#endif