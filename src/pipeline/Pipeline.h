//
// Created by lijin on 2023/12/6.
//

#ifndef TRT_YOLOV8_SERVER_PIPELINE_H
#define TRT_YOLOV8_SERVER_PIPELINE_H

#include "../base/ProcessNode.hpp"
#include "../base/TransferData.h"
#include <atomic>
#include <future>
#include <memory>

namespace pipeline {

/**
 * @brief 流水线基类
 * @details 用于管理node节点，以及节点之间的连接关系
 * 一个pipeline就是一个完成的视频图像处理任务流程，一个任务由多个节点组成
 * 子类需要在init函数中初始化节点和连接关系
 */
class Pipeline {
public:
  typedef std::shared_ptr<Pipeline> ptr;
  //    typedef std::function<void(Data::BaseData::ptr)> DataInputFunc;

protected:
  std::mutex m_mutex;
  std::atomic<bool> m_initialized{false};
  std::string m_task_name;
  std::vector<Base::Node::ptr> m_nodes;

public:
  Pipeline() = delete;

  explicit Pipeline(std::string task_name)
      : m_task_name(std::move(task_name)) {}

  virtual ~Pipeline();

  std::string get_name() const { return m_task_name; }

public:
  /**
   * @brief 启动所有的节点线程
   */
  void Start();

  /**
   * @brief 停止所有的节点线程
   */
  void Stop();

  /**
   * @brief 初始化内部节点和连接关系
   * @return 是否初始化成功
   */
  virtual bool Init() = 0;
};

} // namespace pipeline

#endif // TRT_YOLOV8_SERVER_PIPELINE_H
