//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_YOLOV5DETECTIONINFER_H
#define VIDEOPIPELINE_YOLOV5DETECTIONINFER_H

#include "infer/InferPipeline.h"

namespace infer {

class Yolov5DetectionInfer : InferPipeline {
public:
    typedef std::shared_ptr<Yolov5DetectionInfer> ptr;

    Yolov5DetectionInfer(std::string infer_name,
                         std::string model_path,
                         int         device_id      = 0,
                         int         max_batch_size = 16);

private:
    void pre_process(std::vector<Data::BaseData::ptr> &batch_data) override;
    void post_process(std::vector<Data::BaseData::ptr> &batch_data) override;
    void infer_process(std::vector<Data::BaseData::ptr> &batch_data) override;

private:
};

}  // namespace infer

#endif  // VIDEOPIPELINE_YOLOV5DETECTIONINFER_H
