//
// Created by lijin on 2023/12/19.
//

#ifndef VIDEOPIPELINE_INFERENCEDTO_H
#define VIDEOPIPELINE_INFERENCEDTO_H

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include OATPP_CODEGEN_BEGIN(DTO)

namespace Dto {

/*

 {
   "inference": [
     {
       "inference_name": "推理实例名称",
       "inference_type": "推理类型，/分类/检测/识别/跟踪",
       "model_path": "模型路径",
       "label_path": "标签路径",
       "max_batch_size": "最大批处理大小",
       "device_instance": [
         0
       ]
     }
   ]
 }
 */

class InferenceDto : public oatpp::DTO {
    DTO_INIT(InferenceDto, DTO)
    DTO_FIELD(String, inference_name);
    DTO_FIELD(String, inference_type);
    DTO_FIELD(String, model_path);
    DTO_FIELD(String, label_path);
    DTO_FIELD(Int32, max_batch_size);
    DTO_FIELD(List<Int32>::ObjectWrapper, device_instance);
};

}  // namespace Dto

#include OATPP_CODEGEN_END(DTO)

#endif  // VIDEOPIPELINE_INFERENCEDTO_H
