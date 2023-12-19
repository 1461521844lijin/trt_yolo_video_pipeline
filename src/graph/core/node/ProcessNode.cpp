//
// Created by lijin on 2023/12/18.
//

#include "ProcessNode.h"

namespace GraphCore {

std::string Node::getName() {
    return m_name;
}

void Node::setName(const std::string &name) {
    m_name = name;
}

Node::~Node() {
    Stop();
}
Node::Node(std::string name, NODE_TYPE type) : m_name(std::move(name)), m_type(type) {}

void Node::Start() {
    if (!m_run) {
        m_run = true;
        if (before_start_cb) {
            before_start_cb(getName(), StatusCode::OK, "节点开始线程启动");
        }
        m_worker = std::thread(&Node::worker, this);
        // todo 这里需要等待线程启动完成才能同步执行
        if (after_start_cb) {
            after_start_cb(getName(), StatusCode::OK, "节点线程启动完成");
        }
    } else {
        throw std::runtime_error("改线程重复启动");
    }
}

void Node::Stop() {
    if (!m_run) {
        return;
    }
    m_run = false;
    std::for_each(m_input_buffers.begin(), m_input_buffers.end(),
                  [&](const auto &item) { item.second->clear(); });
    std::for_each(m_output_buffers.begin(), m_output_buffers.end(),
                  [&](const auto &item) { item.second->clear(); });
    m_base_cond->notify_all();
    if (m_worker.joinable())
        m_worker.join();
    if (exit_cb) {
        std::string name = getName();
        exit_cb(getName(), StatusCode::NodeExit, "节点线程退出");
    }
    std::cout << getName() << " exit" << std::endl;
}

void Node::add_input(const std::string &tag, QUEUE queue) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    queue->setCond(m_base_cond);
    m_input_buffers.insert(make_pair(tag, queue));
}

void Node::add_output(const std::string &tag, QUEUE queue) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_output_buffers.insert(make_pair(tag, queue));
}

void Node::del_input(const std::string &tag) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_input_buffers.find(tag) != m_input_buffers.end()) {
        m_input_buffers.erase(tag);
    }
}

void Node::del_output(const std::string &tag) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_output_buffers.find(tag) != m_output_buffers.end()) {
        m_output_buffers.erase(tag);
    }
}

void Node::worker() {
    std::vector<BaseData::ptr> datas;
    while (m_run) {
        get_input_data(datas);
        if (!datas.empty()) {
            for (auto &data : datas) {
                data = handle_data(data);
                send_output_data(data);
            }
        } else {
            std::unique_lock<std::mutex> lk(m_base_mutex);
            m_base_cond->wait(lk);
            continue;
        }
    }
}

void Node::get_input_data(std::vector<BaseData::ptr> &datas, int max_size) {
    datas.clear();
    for (auto &item : m_input_buffers) {
        BaseData::ptr data;
        int           count = 0;
        while (item.second->Pop(data)) {
            datas.push_back(data);
            count++;
            if (count >= max_size) {
                break;
            }
        }
    }
}

void Node::send_output_data(const BaseData::ptr &data) {
    if (!data) {
        return;
    }
    for (auto &item : m_output_buffers) {
        if (!item.second->Push(data)) {
            if (buffer_over_cb) {
                buffer_over_cb(getName(), StatusCode::NodeBufferOver, item.first + "缓冲队列已满");
            }
        }
    }
}
void Node::send_output_datas(const std::vector<BaseData::ptr> &datas) {
    for (auto &data : datas) {
        send_output_data(data);
    }
}

BaseData::ptr Node::handle_data(BaseData::ptr data) {
    return data;
}

void Node::add_data(const BaseData::ptr &data) {
    m_input_buffers.begin()->second->push_front(data);
}
void Node::add_datas(const std::vector<BaseData::ptr> &datas) {
    for (auto &data : datas) {
        add_data(data);
    }
}
NODE_TYPE Node::getType() {
    return MID_NODE;
}

}  // namespace GraphCore