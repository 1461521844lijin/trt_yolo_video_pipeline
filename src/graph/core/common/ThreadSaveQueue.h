//
// Created by lijin on 2023/12/18.
//

#ifndef TRT_YOLOV8_THREADSAVEQUEUE_H
#define TRT_YOLOV8_THREADSAVEQUEUE_H

#include "graph/core/common/BaseData.h"
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>

namespace GraphCore {

// 缓冲队列满时的策略
enum BufferOverStrategy {
    DROP_EARLY,  // 丢弃最早的帧
    DROP_LATE,   // 丢弃最新的帧
    CLEAR,       // 清空缓冲队列
    BLOCK        // 堵塞，直到队列有空间
};

// 线程安全队列
class ThreadSaveQueue {
public:
    typedef std::shared_ptr<ThreadSaveQueue> ptr;

public:
    // 像队列中添加数据，返回true表示添加成功，返回false表示队列满，缓冲中有数据被丢弃，但该帧数据被添加成功
    bool Push(const Data::BaseData::ptr &data) {
        if (m_list.size() > max_number) {
            switch (m_buffer_strategy) {
                case BufferOverStrategy::DROP_EARLY: {
                    // 缓存队列满了，丢弃最早的帧，保证实时性，但不丢弃其他信息数据
                    if (m_list.front()->get_data_type() == Data::DataType::FRAME) {
                        {
                            std::unique_lock<std::mutex> lock(m_mutex);
                            m_list.pop_front();
                            m_list.push_back(data);
                        }
                        m_work_cond->notify_one();
                        return true;  // 丢弃的是最早的帧, 但是添加成功
                    }
                    break;
                }
                case BufferOverStrategy::DROP_LATE: {
                    if (m_list.front()->get_data_type() == Data::DataType::FRAME) {
                        m_work_cond->notify_one();
                        return false;  // 丢弃的是最新的帧
                    }
                    ErrorL << "缓冲队列满，丢弃最新的帧";
                    break;
                }
                case BufferOverStrategy::CLEAR: {
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_list.clear();
                    }
                    DebugL << "清空缓冲队列，丢弃所有数据，重新添加数据";
                    m_work_cond->notify_one();
                    return false;
                }
                case BufferOverStrategy::BLOCK: {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_self_cond.wait(lock);
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_list.push_back(data);
                        return true;
                    }
                    break;
                }
                default: {
                    throw std::runtime_error("unknown buffer over strategy");
                }
            }
        } else {
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_list.push_back(data);
            }
            m_work_cond->notify_one();
            return true;
        }
        return false;
    }

    bool Pop(Data::BaseData::ptr &data) {
        if (m_list.empty()) {
            return false;
        }
        // 尝试尽量减少临界区的代码量
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            data = m_list.front();
            m_list.pop_front();
        }
        if (m_buffer_strategy == BufferOverStrategy::BLOCK) {
            m_self_cond.notify_one();
        }
        return true;
    }

    // PopList函数，从队列中取出最多num个数据
    bool PopList(std::vector<Data::BaseData::ptr> &data_list, int max_num) {
        if (m_list.empty()) {
            return false;
        }
        // 尝试尽量减少临界区的代码量
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            for (int i = 0; i < max_num; i++) {
                if (m_list.empty()) {
                    break;
                }
                auto data = m_list.front();
                data_list.push_back(data);
                m_list.pop_front();
//                ErrorL <<"拿出队列：" <<data->Get<FRAME_INDEX_TYPE>(FRAME_INDEX);
            }
        }
        if (m_buffer_strategy == BufferOverStrategy::BLOCK) {
            m_self_cond.notify_one();
        }
        return true;
    }




    void push_front(const Data::BaseData::ptr &data) {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_list.push_front(data);
        }
        m_work_cond->notify_one();
    }

    void set_max_size(const int size) {
        std::unique_lock<std::mutex> lock(m_mutex);
        max_number = size;
    }

    int size() {
        std::unique_lock<std::mutex> lock(m_mutex);
        return (int)m_list.size();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_list.clear();
    }

    void setCond(std::shared_ptr<std::condition_variable> &cond) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_work_cond = cond;
    }

    void set_buffer_strategy(BufferOverStrategy strategy) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_buffer_strategy = strategy;
    }

private:
    std::mutex                               m_mutex;
    std::shared_ptr<std::condition_variable> m_work_cond;  // 用于唤醒工作线程的条件变量
    std::condition_variable                  m_self_cond;      // 用于唤醒自身的条件变量
    std::list<Data::BaseData::ptr>           m_list;           // 缓冲队列
    int                                      max_number = 50;  // 默认最大缓冲帧数
    BufferOverStrategy m_buffer_strategy = BufferOverStrategy::DROP_LATE;  // 缓冲队列满时的策略
};

}  // namespace GraphCore

#endif  // TRT_YOLOV8_THREADSAVEQUEUE_H
