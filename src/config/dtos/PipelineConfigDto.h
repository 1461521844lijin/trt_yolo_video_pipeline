//
// Created by lijin on 2023/12/19.
//

#ifndef VIDEOPIPELINE_PIPELINECONFIGDTO_H
#define VIDEOPIPELINE_PIPELINECONFIGDTO_H

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include "InferenceDto.h"
#include "PipelineDto.h"

/*
{
  "inferences": [
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
  ],
  "pipelines": [
    {
      "pipeline_name": "流水线名称",
      "nodes": [
        {
          "node_name": "节点名称",
          "node_type": "节点类型，/输入/中间/推理/输出",
          "node_params": {
            "input_addr": "数据源地址, 可以是视频流地址，或mp4文件地址",
            "inference_name": "推理实例名称",
            "hw_encode": "是否使用硬件编码，true/false",
            "hw_decode": "是否使用硬件解码，true/false",
            "output_addr": "输出地址，可以是rtmp地址，或mp4文件地址",
            "output_bitrate": "输出码率",
            "output_width": "输出宽度",
            "output_height": "输出高度"
          },
          "last_node": "上一个节点名称",
          "next_node": "下一个节点名称"
        }
      ]
    }
  ]
}

*/

#include OATPP_CODEGEN_BEGIN(DTO)

namespace Dto {

class PipelineConfigDto : public oatpp::DTO {
    DTO_INIT(PipelineConfigDto, DTO)
    DTO_FIELD(List<Object<InferenceDto>::ObjectWrapper>::ObjectWrapper, inferences);
    DTO_FIELD(List<Object<PipelineDto>::ObjectWrapper>::ObjectWrapper, pipelines);
};

}  // namespace Dto
#include OATPP_CODEGEN_END(DTO)
#endif  // VIDEOPIPELINE_PIPELINECONFIGDTO_H
