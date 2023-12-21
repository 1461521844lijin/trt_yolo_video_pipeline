//
// Created by lijin on 2023/12/21.
//

#include "Yolov5DetectionInfer.h"

#include <utility>

namespace infer {
Yolov5DetectionInfer::Yolov5DetectionInfer(std::string infer_name,
                                           std::string model_path,
                                           int         device_id,
                                           int         max_batch_size)
    : InferPipeline(std::move(infer_name), std::move(model_path), device_id, max_batch_size) {}
void Yolov5DetectionInfer::pre_process(std::vector<Data::BaseData::ptr> &batch_data) {
    InferPipeline::pre_process(batch_data);
}
void Yolov5DetectionInfer::post_process(std::vector<Data::BaseData::ptr> &batch_data) {
    InferPipeline::post_process(batch_data);
}
void Yolov5DetectionInfer::infer_process(std::vector<Data::BaseData::ptr> &batch_data) {
    InferPipeline::infer_process(batch_data);
}
}  // namespace infer