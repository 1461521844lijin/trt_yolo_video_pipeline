#pragma once
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace utils {

template <typename T>
class ThreadsafeQueue {
private:
    std::queue<T>           m_queue;
    mutable std::mutex      m_mutex;
    std::condition_variable m_cv;
    int                     m_max_size = 0;
    bool                    m_running  = true;

public:
    static inline std::shared_ptr<ThreadsafeQueue> createShared() {
        return std::make_shared<ThreadsafeQueue>();
    }

    static inline std::shared_ptr<ThreadsafeQueue> createShared(int max_size) {
        return std::make_shared<ThreadsafeQueue>(max_size);
    }

public:
    ThreadsafeQueue() {}
    ThreadsafeQueue(int max_size) {
        m_max_size = max_size;
    }

    /*!
     * @brief 向队列中添加一个元素
     * @param value 要添加的元素
     */
    void push(T value) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_max_size > 0 && m_queue.size() >= m_max_size) {
            m_queue.pop();
        }
        m_queue.push(value);
        m_cv.notify_one();
    }

    /*!
     * @brief 尝试从队列中取出一个元素
     * @param value 取出的元素
     * @return 如果队列为空，返回false，否则返回true
     */
    bool try_pop(T &value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty())
            return false;

        value = m_queue.front();
        m_queue.pop();
        return true;
    }

    /*!
     * @brief 从队列中取出一个元素，如果队列为空，则阻塞
     * @param value 取出的元素
     * @return 如果队列退出，返回false并且无结果，否则返回true
     */
    bool wait_and_pop(T &value) {
        std::unique_lock<std::mutex> lock(m_mutex);

        // m_cv.wait(lock, [this]{ return !m_queue.empty(); });
        while (m_queue.empty()) {
            if (m_running == false)
                return false;
            m_cv.wait(lock);
        }

        value = m_queue.front();
        m_queue.pop();
        return true;
    }

    /*!
     * @brief 从队列中取出一个元素，如果队列为空，则阻塞
     * @param value 取出的元素
     * @param timeout 超时时间，单位毫秒
     */
    bool wait_and_pop(T &value, int timeout) {
        std::unique_lock<std::mutex> lock(m_mutex);

        // m_cv.wait(lock, [this]{ return !m_queue.empty(); });
        while (m_queue.empty()) {
            if (m_cv.wait_for(lock, std::chrono::milliseconds(timeout)) ==
                    std::cv_status::timeout ||
                m_running == false) {
                return false;
            }
        }

        value = m_queue.front();
        m_queue.pop();
        return true;
    }

    /*!
     * @brief 清空队列
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }

    int size() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

    void remove_front() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty())
            return;
        m_queue.pop();
        return;
    }

    void wait_front(T &value) {
        std::unique_lock<std::mutex> lock(m_mutex);

        m_cv.wait(lock, [this] { return !m_queue.empty(); });

        value = m_queue.front();
    }

    bool try_front(T &value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty())
            return false;

        value = m_queue.front();
        return true;
    }

    void exit() {
        m_running = false;
        m_cv.notify_all();
    }

    ~ThreadsafeQueue() {
        exit();
    }
};

}  // namespace utils