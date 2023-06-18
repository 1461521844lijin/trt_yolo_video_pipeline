#pragma once

#include "TransferData.h"
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <list>

namespace Base {

using namespace std;

// 线程安全队列
template <class T>
class SList {
public:
    typedef std::shared_ptr<SList> ptr;

public:
    bool Push(T data) {
        unique_lock<mutex> lock(m_mutex);
        m_list.push_back(data);
        m_cond->notify_one();
        if (m_list.size() > max_number) {
            m_list.pop_front();
            return false;
        }
        return true;
    }

    bool Pop(T &data) {
        unique_lock<mutex> lock(m_mutex);
        if (m_list.size() == 0)
            return false;
        data = m_list.front();
        m_list.pop_front();
        return true;
    }

    void set_max_size(const int size) {
        unique_lock<mutex> lock(m_mutex);
        max_number = size;
    }

    int size() {
        return m_list.size();
    }

    void clear() {
        unique_lock<mutex> lock(m_mutex);
        m_list.clear();
    }
    void setCond(std::shared_ptr<std::condition_variable> &cond) {
        m_cond = cond;
    }

private:
    std::mutex                               m_mutex;
    std::shared_ptr<std::condition_variable> m_cond;
    list<T>                                  m_list;
    int                                      max_number = 25;
};


class Node {
public:
    typedef std::shared_ptr<Node>                         ptr;
    typedef SList<std::shared_ptr<Data::Input_Data>>::ptr QUEUE;
    Node();
    Node(const std::string &name) : m_name(name) {}
    virtual ~Node(){};

public:
    virtual void Start();
    virtual void Stop();
    std::string  getName();
    void         setName(const std::string &name);

    virtual void add_input(const std::string &tag, QUEUE queue);
    virtual void add_output(const std::string &tag, QUEUE queue);
    virtual void del_input(const std::string &tag);
    virtual void del_output(const std::string &tag);

    void add_input_data(std::shared_ptr<Data::Input_Data> data);

protected:
    virtual void worker() = 0;

protected:
    std::string                              m_name;
    std::thread                              m_worker;
    bool                                     m_run = false;
    std::mutex                               m_base_mutex;
    std::shared_ptr<std::condition_variable> m_base_cond = std::make_shared<std::condition_variable>();

    std::map<std::string, QUEUE> m_input_buffers;
    std::map<std::string, QUEUE> m_output_buffers;
};

static inline void LinkNode(Node::ptr front, Node::ptr back, int max_cache = 25) {
    auto queue = std::make_shared<SList<std::shared_ptr<Data::Input_Data>>>();
    queue->set_max_size(max_cache);
    front->add_output(back->getName(), queue);
    back->add_input(front->getName(), queue);
}

static inline void UnLinkNode(Node::ptr front, Node::ptr back) {
    front->del_output(back->getName());
    back->del_input(front->getName());
}

}  // namespace Base
