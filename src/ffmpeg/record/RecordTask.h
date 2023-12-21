//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_RECORDTASK_H
#define VIDEOPIPELINE_RECORDTASK_H

#include "graph/core/common/BaseData.h"
#include "utils/ThreadsafeQueue.h"
#include <functional>
#include <memory>
#include <string>

namespace record {

enum RecordType {
    IMAGE_RECORD,  // 图片录制
    VIDEO_RECORD   // 视频录制
};

enum RecordStatus {
    NO_START,  // 未开始
    STARTING,  // 进行中
    COMPLETED  // 已完成
};

struct RecordConfig {
    std::string save_path;                  // 保存路径
    std::string file_name;                  // 文件名
    RecordType  record_type;                // 录制类型
    int         duration;                   // 录制时长
    int         src_width;                  // 宽度
    int         src_height;                 // 高度
    int         dst_width;                  // 宽度
    int         dst_height;                 // 高度
    int         fps     = 25;               // 帧率
    int         bitrate = 2 * 1024 * 1024;  // 码率
};

// 视频录制任务，任务启动会创建一个线程，线程中不断从队列中取数据，然后写入文件
class RecordTask : public std::enable_shared_from_this<RecordTask> {
public:
    typedef std::shared_ptr<RecordTask>                         ptr;
    typedef std::function<void(std::string, int, RecordConfig)> RecordEvent_CB;

protected:
    RecordConfig   m_config;                                // 录制配置
    RecordStatus   record_status = RecordStatus::NO_START;  // 录制状态
    RecordEvent_CB record_complete_cb;                      // 录制结束回调
    int64_t        m_create_time;                           // 任务创建时间

    std::thread                                 m_thread;   // 工作线程
    utils::ThreadsafeQueue<Data::BaseData::ptr> m_queue;    // 数据队列
    std::mutex                                  m_mutex;    // 互斥锁
    std::condition_variable                     m_cond;     // 条件变量
protected:
    virtual void worker();

    virtual void record_handler(Data::BaseData::ptr data) = 0;

public:
    RecordTask() = delete;

    explicit RecordTask(RecordConfig config);

    virtual ~RecordTask();

    void Start();

    void Stop();

    void set_record_complete_cb(RecordEvent_CB cb);

    RecordStatus get_record_status();

    void push(Data::BaseData::ptr data);
};

}  // namespace record

#endif  // VIDEOPIPELINE_RECORDTASK_H
