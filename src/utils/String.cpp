#include "String.h"

namespace utils {
std::string String::UrlDecode(const std::string &str) {
    std::string strTemp;
    size_t      length = str.length();
    for (size_t i = 0; i < length; i++) {
        if (str[i] == '+')
            strTemp += ' ';
        else if (str[i] == '%') {
            if (i + 2 >= length)
                break;
            unsigned char high = FromHex((unsigned char)str[++i]);
            unsigned char low  = FromHex((unsigned char)str[++i]);
            strTemp += high * 16 + low;
        } else
            strTemp += str[i];
    }
    return strTemp;
}

std::string String::UrlEncode(const std::string &str) {
    std::string strTemp;
    size_t      length = str.length();
    for (size_t i = 0; i < length; i++) {
        if (isalnum((unsigned char)str[i]) || (str[i] == '-') || (str[i] == '_') ||
            (str[i] == '.') || (str[i] == '~'))
            strTemp += str[i];
        else if (str[i] == ' ')
            strTemp += "+";
        else {
            strTemp += '%';
            strTemp += ToHex((unsigned char)str[i] >> 4);
            strTemp += ToHex((unsigned char)str[i] % 16);
        }
    }
    return strTemp;
}

unsigned char String::FromHex(unsigned char x) {
    unsigned char y;
    if (x >= 'A' && x <= 'Z')
        y = x - 'A' + 10;
    else if (x >= 'a' && x <= 'z')
        y = x - 'a' + 10;
    else if (x >= '0' && x <= '9')
        y = x - '0';
    else
        return 0;
    return y;
}

unsigned char String::ToHex(unsigned char x) {
    return x > 9 ? x + 55 : x + 48;
}

std::string String::Time2Str(time_t ts, const std::string &format) {
    struct tm tm {};
    localtime_r(&ts, &tm);
    char buf[64];
    strftime(buf, sizeof(buf), format.c_str(), &tm);
    return buf;
}

time_t String::Str2Time(const char *str, const char *format) {
    struct tm t {};
    memset(&t, 0, sizeof(t));
    if (!strptime(str, format, &t)) {
        return 0;
    }
    return mktime(&t);
}

std::vector<std::string> String::split(const std::string &s, const char *delim) {
    std::vector<std::string> ret;
    size_t                   last  = 0;
    auto                     index = s.find(delim, last);
    while (index != std::string::npos) {
        if (index - last > 0) {
            ret.push_back(s.substr(last, index - last));
        }
        last  = index + strlen(delim);
        index = s.find(delim, last);
    }
    if (!s.size() || s.size() - last > 0) {
        ret.push_back(s.substr(last));
    }
    return ret;
}

bool String::start_with(const std::string &str, const std::string &substr) {
    return str.find(substr) == 0;
}

}  // namespace utils