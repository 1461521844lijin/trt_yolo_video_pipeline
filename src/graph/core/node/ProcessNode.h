//
// Created by lijin on 2023/12/18.
//

#ifndef VIDEOPIPELINE_PROCESSNODE_H
#define VIDEOPIPELINE_PROCESSNODE_H

#include "graph/core/common/IDataHooker.h"
#include "graph/core/common/IEventCallback.h"
#include "graph/core/common/StatusCode.h"
#include "graph/core/common/ThreadSaveQueue.h"
#include <condition_variable>
#include <functional>
#include <iostream>
#include <map>

namespace GraphCore {

enum NODE_TYPE {
    SRC_NODE,  // 输入节点
    MID_NODE,  // 中间节点
    DES_NODE,  // 输出节点
};

class Node : public IEventCallBack, public IDataHooker {
public:
    typedef std::shared_ptr<Node> ptr;
    typedef ThreadSaveQueue::ptr  QUEUE;

    Node() = delete;

    explicit Node(std::string name, NODE_TYPE type);

    virtual ~Node();

public:
    /**
     * @brief 启动工作线程
     */
    void Start();

    /**
     * @brief 停止线程
     */
    void Stop();

    std::string getName();

    NODE_TYPE getType();

    void setName(const std::string &name);

    virtual void add_input(const std::string &tag, QUEUE queue);

    /**
     * @brief
     * 给当前节点添加输入缓冲队列，所有的输入缓冲都共享该同一个条件变量，因此N个输入节点的任意数据输入都会唤醒当前堵塞的worker线程
     * @param tag
     * @param queue
     */
    virtual void add_output(const std::string &tag, QUEUE queue);

    virtual void del_input(const std::string &tag);

    virtual void del_output(const std::string &tag);

    /**
     * @brief 向节点的输入队列中添加数据，主要是用来输入控制和配置信息
     * @param data
     */
    void add_data(const Data::BaseData::ptr &data);

    void add_datas(const std::vector<Data::BaseData::ptr> &datas);

    void set_get_data_max_num(int num);

protected:
    virtual void worker();

    /**
     * @brief 从输入队列中获取数据，如果队列为空则阻塞等待
     * @param datas
     */
    virtual void get_input_datas(std::vector<Data::BaseData::ptr> &datas);

    virtual void send_output_data(const Data::BaseData::ptr &data);

    void send_output_datas(const std::vector<Data::BaseData::ptr> &datas);

    /**
     * @brief 处理数据业务接口
     * @param data 所有数据的继承基类
     * @return
     */
    virtual Data::BaseData::ptr handle_data(Data::BaseData::ptr data);

protected:
    std::string                              m_name;
    std::thread                              m_worker;
    NODE_TYPE                                m_type;
    bool                                     m_run              = false;
    int                                      m_get_data_max_num = 1;
    std::mutex                               m_base_mutex;
    std::shared_ptr<std::condition_variable> m_base_cond =
        std::make_shared<std::condition_variable>();

    std::map<std::string, QUEUE> m_input_buffers;
    std::map<std::string, QUEUE> m_output_buffers;
};

/**
 * @brief 将两个节点连接起来
 * @param front  前一个节点
 * @param back   后一个节点
 * @param max_cache  缓冲队列最大缓存帧数,默认25
 * @param strategy   缓冲队列满时的策略，默认丢弃最早的帧
 */
static inline void LinkNode(const Node::ptr   &front,
                            const Node::ptr   &back,
                            int                max_cache = 25,
                            BufferOverStrategy strategy  = BufferOverStrategy::DROP_EARLY) {
    auto queue = std::make_shared<ThreadSaveQueue>();
    queue->set_max_size(max_cache);
    queue->set_buffer_strategy(strategy);
    back->add_input(front->getName(), queue);
    front->add_output(back->getName(), queue);
}

static inline void UnLinkNode(const Node::ptr &front, const Node::ptr &back) {
    front->del_output(back->getName());
    back->del_input(front->getName());
}

}  // namespace GraphCore

#endif  // VIDEOPIPELINE_PROCESSNODE_H
