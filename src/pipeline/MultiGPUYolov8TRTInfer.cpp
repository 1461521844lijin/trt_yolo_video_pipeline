//
// Created by lijin on 2023/12/6.
//

#include "MultiGPUYolov8TRTInfer.h"

#include <utility>

namespace infer {
MultiGPUYolov8TRTInfer::MultiGPUYolov8TRTInfer(
    const std::string &name, const std::string &engine_path, int max_batch_size,
    std::vector<int> device_id_list, yolo::Type yolo_type,
    float confidence_threshold, float nms_threshold)
    : m_name(name), m_engine_path(engine_path),
      m_max_batch_size(max_batch_size),
      m_device_id_list(std::move(device_id_list)) {
  for (int device_id : m_device_id_list) {
    auto trt_instance = std::make_shared<
        cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>>();
    trt_instance->start(
        [&] {
          return yolo::load(m_engine_path, yolo_type, confidence_threshold,
                            nms_threshold, device_id);
        },
        m_max_batch_size);
    m_infer_list.push_back(trt_instance);
  }
}

std::string MultiGPUYolov8TRTInfer::get_name() const { return m_name; }

int MultiGPUYolov8TRTInfer::get_infer_index() {
  m_infer_index = (m_infer_index + 1) % m_infer_list.size();
  return m_infer_index;
}
std::shared_future<yolo::BoxArray>
MultiGPUYolov8TRTInfer::commit(const yolo::Image &input) {
  return m_infer_list[get_infer_index()]->commit(input);
}

} // namespace infer
