#include "trt_node.hpp"
#include "base/TransferData.h"
#include <iostream>

namespace trt
{
    void TrtNode::set_trt_instance(TRTInstancePtr &instance){
        m_trt_instance = instance;
    }

    void TrtNode::worker()
    {
        std::shared_ptr<Data::Input_Data> data;
        while (m_run)
        {
            for (auto &input : m_input_buffers)
            {
                if (input.second->Pop(data))
                {
                    std::shared_ptr<Data::Decode_Data> decode_data =
                        std::dynamic_pointer_cast<Data::Decode_Data>(data);
                        yolo::Image image(decode_data->original_image.data, decode_data->original_image.cols, decode_data->original_image.rows);
                    decode_data->boxarray_fu = m_trt_instance->commit(image);
                    for (auto &output : m_output_buffers)
                    {
                        output.second->Push(decode_data);
                    }
                }else{
                    if(!m_run) break;
                    std::unique_lock<std::mutex> lk(m_base_mutex);
                    m_base_cond->wait(lk);
                }
            }
        }
    }
}