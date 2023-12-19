//
// Created by lijin on 2023/12/19.
//

#ifndef VIDEOPIPELINE_PIPELINEDTO_H
#define VIDEOPIPELINE_PIPELINEDTO_H

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include OATPP_CODEGEN_BEGIN(DTO)

/*

 {
   "pipeline": {
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
 }
 */

namespace Dto {
using namespace oatpp;

class NodeParamsDto : public oatpp::DTO {
    DTO_INIT(NodeParamsDto, DTO)
    DTO_FIELD(String, input_addr)     = "";
    DTO_FIELD(String, inference_name) = "";
    DTO_FIELD(Boolean, hw_encode)     = "";
    DTO_FIELD(Boolean, hw_decode)     = "";
    DTO_FIELD(String, output_addr)    = "";
    DTO_FIELD(Int32, output_bitrate)  = -1;
    DTO_FIELD(Int32, output_width)    = -1;
    DTO_FIELD(Int32, output_height)   = -1;
};

class NodeDto : public oatpp::DTO {
    DTO_INIT(NodeDto, DTO)
    DTO_FIELD(String, node_name) = "";
    DTO_FIELD(String, node_type) = "";
    DTO_FIELD(Object<NodeParamsDto>, node_params);
    DTO_FIELD(String, last_node) = "";
    DTO_FIELD(String, next_node) = "";
};

class PipelineDto : public oatpp::DTO {
    DTO_INIT(PipelineDto, DTO)
    DTO_FIELD(String, pipeline_name) = "";
    DTO_FIELD(List<Object<NodeDto>>, nodes);
};

}  // namespace Dto
#include OATPP_CODEGEN_END(DTO)
#endif  // VIDEOPIPELINE_PIPELINEDTO_H
