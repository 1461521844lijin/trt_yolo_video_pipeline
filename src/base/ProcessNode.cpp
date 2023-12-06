#include "ProcessNode.hpp"
#include <iostream>

namespace Base {

std::string Node::getName() {
    return m_name;
}
void Node::setName(const std::string &name) {
    m_name = name;
}

void Node::Start() {
    if (!m_run) {
        m_run    = true;
        m_worker = std::thread(&Node::worker, this);
    } else {
        throw std::runtime_error("改线程重复启动");
    }
}

void Node::Stop() {
    if (!m_run) {
        return;
    }
    m_run = false;
    m_base_cond->notify_all();
    if (m_worker.joinable())
        m_worker.join();
    std::cout << getName() + " 节点退出" << std::endl;
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

void Node::add_input_data(std::shared_ptr<Data::Input_Data> data) {
    if(m_input_buffers.size() == 0){
        m_input_buffers["default_input"] =  std::make_shared<SList<std::shared_ptr<Data::Input_Data>>>();
        m_input_buffers["default_input"]->setCond(m_base_cond);
    }
    for (auto input : m_input_buffers) {
        input.second->Push(data);
        break;
    }
}

}  // namespace Base
