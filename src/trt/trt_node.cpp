#include "trt_node.hpp"
#include "graph/core/common/BaseData.h"
#include <iostream>

namespace trt
{
    void TrtNode::set_trt_instance(TRTInstancePtr &instance){
        m_trt_instance = instance;
    }

    void TrtNode::worker()
    {
        std::shared_ptr<Data::BaseData> data;
        while (m_run)
        {
            for (auto &input : m_input_buffers)
            {
                if (input.second->Pop(data))
                {
                }else{
                    if(!m_run) break;
                    std::unique_lock<std::mutex> lk(m_base_mutex);
                    m_base_cond->wait(lk);
                }
            }
        }
    }
}