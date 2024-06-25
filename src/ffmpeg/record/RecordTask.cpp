//
// Created by lijin on 2023/12/21.
//

#include "RecordTask.h"
#include <utility>

namespace record {
RecordTask::RecordTask(RecordConfig config) : m_config(std::move(config)) {
    m_create_time = time(nullptr);
}

RecordTask::~RecordTask() {
    Stop();
}

void RecordTask::Start() {
    if (record_status != RecordStatus::NO_START) {
        return;
    }
    record_status = RecordStatus::STARTING;
    m_thread      = std::thread(&RecordTask::worker, this);
}

void RecordTask::worker() {
    while (record_status == RecordStatus::STARTING) {
        Data::BaseData::ptr data;
        if (m_queue.try_pop(data)) {
            record_handler(data);
        } else {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cond.wait(lock);
        }
    }
}

void RecordTask::Stop() {
    record_status = RecordStatus::COMPLETED;
    m_cond.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }
    std::cout << "RecordTask Stop" << std::endl;
}

void RecordTask::set_record_complete_cb(RecordTask::RecordEvent_CB cb) {
    record_complete_cb = std::move(cb);
}

RecordStatus RecordTask::get_record_status() {
    return record_status;
}

void RecordTask::push(Data::BaseData::ptr data) {
    m_queue.push(std::move(data));
    m_cond.notify_one();
}

}  // namespace record
   // record