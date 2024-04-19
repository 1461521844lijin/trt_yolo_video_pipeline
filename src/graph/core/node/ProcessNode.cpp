//
// Created by lijin on 2023/12/18.
//

#include "ProcessNode.h"

#include <utility>

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
            if (before_start_cb(getName(), StatusCode::OK, "节点开始线程启动") != 0) {
                throw std::runtime_error("节点初始化失败");
            }
        }
        m_worker = std::thread(&Node::worker, this);
        // todo 这里需要等待线程启动完成才能同步执行
        if (after_start_cb) {
            after_start_cb(getName(), StatusCode::OK, "节点线程启动完成");
        }
    } else {
        throw std::runtime_error("线程重复启动");
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
    std::vector<Data::BaseData::ptr> datas;
    while (m_run) {
        get_input_datas(datas);
        if (!datas.empty()) {
            if (batch_data_handler_hooker) {
                auto res = batch_data_handler_hooker(datas);
                send_output_datas(res);
                continue;
            }
            for (auto &data : datas) {
                if (data_handler_hooker) {
                    data = data_handler_hooker(data);
                } else {
                    data = handle_data(data);
                }
                send_output_data(data);
            }
        } else {
            std::unique_lock<std::mutex> lk(m_base_mutex);
            m_base_cond->wait(lk);
            continue;
        }

    }

    
}

void Node::get_input_datas(std::vector<Data::BaseData::ptr> &datas) {
    datas.clear();
    for (auto &item : m_input_buffers) {
        Data::BaseData::ptr data;
        int                 count = 0;
        while (item.second->Pop(data)) {
            datas.push_back(data);
            count++;
            if (count >= m_get_data_max_num) {
                break;
            }
        }
    }
}

void Node::send_output_data(const Data::BaseData::ptr &data) {
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
void Node::send_output_datas(const std::vector<Data::BaseData::ptr> &datas) {
    for (auto &data : datas) {
        send_output_data(data);
    }
}

Data::BaseData::ptr Node::handle_data(Data::BaseData::ptr data) {
    return data;
}


bool Node::add_data_back(const Data::BaseData::ptr &data){
    return m_input_buffers.begin()->second->Push(data);
}


void Node::add_data(const Data::BaseData::ptr &data) {
    m_input_buffers.begin()->second->push_front(data);
}
void Node::add_datas(const std::vector<Data::BaseData::ptr> &datas) {
    for (auto &data : datas) {
        add_data(data);
    }
}

NODE_TYPE Node::getType() {
    return MID_NODE;
}

void Node::set_get_data_max_num(int num) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_get_data_max_num = num;
}

void Node::set_extra_input_callback(Node::ExtraInputCallBack callback) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_extra_input_callback = std::move(callback);
}

void Node::add_extra_data(const Data::BaseData::ptr &data) {
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_extra_input_callback) {
        m_extra_input_callback(data);
    }
}
bool Node::Init() {
    return true;
}

}  // namespace GraphCore