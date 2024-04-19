#pragma once
#include "chrono"
#include <iostream>

namespace utils {

#define TICKER TICKER1(0)
#define TICKER1(min_ms) TICKER2(min_ms, false)
#define TICKER2(min_ms, print_log)                                                                 \
    utils::Ticker _ticker(__FILE__ + std::string(":") + std::to_string(__LINE__), min_ms, print_log)

class Ticker {
public:
    /**
     * 此对象可以用于代码执行时间统计，以可以用于一般计时
     * @param min_ms 开启码执行时间统计时，如果代码执行耗时超过该参数，则打印警告日志
     * @param print_log  开启ticker不超时时的日志打印
     */
    Ticker(std::string name, double min_ms = 0, bool print_log = false) {
        // 分割name的/，取最后一个
        auto pos = name.find_last_of('/');
        if (pos != std::string::npos) {
            name = name.substr(pos + 1);
        }
        _print_log = print_log;
        _min_ms    = min_ms;
        _name      = name;
        _created = _begin = std::chrono::system_clock::now();
    }

    /*!
     * @brief 获取时间间隔
     * @param time1  开始时间
     * @param time2  结束时间
     * @return  时间间隔，单位毫秒
     */
    static double GetSpan(std::chrono::system_clock::time_point time1,
                          std::chrono::system_clock::time_point time2) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1);
        return double(duration.count()) * std::chrono::microseconds::period::num /
               std::chrono::microseconds::period::den * 1000;
    };

    ~Ticker() {
        double tm = createdTime();
        if (tm > _min_ms) {
            std::cout << "\033[43m\t\t\033[0m" << _name << " time: " << tm << "ms"
                      << ", is over set time" << std::endl;
        } else if (_print_log) {
            std::cout << "\033[42m\t\t\033[0m" << _name << " time: " << tm << "ms" << std::endl;
        }
    }

    /**
     * 获取上次resetTime后至今的时间，单位毫秒
     */
    double elapsedTime() const {
        return GetSpan(_begin, std::chrono::system_clock::now());
    }

    /**
     * 获取从创建至今的时间，单位毫秒
     */
    double createdTime() const {
        return GetSpan(_created, std::chrono::system_clock::now());
    }

    /**
     * 重置计时器
     */
    void resetTime() {
        _begin = std::chrono::system_clock::now();
    }

private:
    bool                                  _print_log;
    double                                _min_ms;
    std::chrono::system_clock::time_point _begin;
    std::chrono::system_clock::time_point _created;
    std::string                           _name;
};

}  // namespace utils
