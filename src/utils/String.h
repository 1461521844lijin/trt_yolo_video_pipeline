#pragma once
#include <cstring>
#include <string>
#include <vector>

namespace utils {

class String {
private:
    static unsigned char ToHex(unsigned char x);

    static unsigned char FromHex(unsigned char x);

public:
    /*!
     * @brief URL编码
     * @param str
     * @return
     */
    static std::string UrlEncode(const std::string &str);

    /*!
     * @brief URL解码
     * @param str
     * @return
     */
    static std::string UrlDecode(const std::string &str);

    /*!
     * @brief 时间戳转字符串
     * @param ts
     * @param format
     * @return
     */
    static std::string Time2Str(time_t             ts     = time(0),
                                const std::string &format = "%Y-%m-%d-%H:%M:%S");

    /*!
     * @brief 字符串转时间戳
     * @param str
     * @param format
     * @return
     */
    static time_t Str2Time(const char *str, const char *format);

    /*!
     * @brief 字符串分割
     * @param s 字符串
     * @param delim 分隔符
     * @return
     */
    static std::vector<std::string> split(const std::string &s, const char *delim);

    // 字符串是否以xx开头
    static bool start_with(const std::string &str, const std::string &substr);
};

}  // namespace utils
