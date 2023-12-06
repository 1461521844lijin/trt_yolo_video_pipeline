//
// Created by lijin on 2023/12/6.
//

#ifndef TRT_YOLOV8_SERVER_MULTIGPUYOLOV8TRTINFER_H
#define TRT_YOLOV8_SERVER_MULTIGPUYOLOV8TRTINFER_H

#include "trt/cpm.hpp"
#include "trt/infer.hpp"
#include "trt/yolo.hpp"

namespace infer {

/*
 * 根据输入的device_id_list，创建对应数量的Yolov8TRTInfer实例，每次commit时，轮流使用其中一个实例，实现多GPU并行推理
 * device_id的值为GPU的索引，例如有两块GPU，device_id_list为[0,1]，则会创建两个Yolov8TRTInfer实例，分别使用GPU0和GPU1
 * 也可以只使用一块GPU，此时device_id_list为[0, 0,
 * 0]，则会创建三个Yolov8TRTInfer实例，均使用GPU0
 */
class MultiGPUYolov8TRTInfer
    : public cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> {
public:
  typedef std::shared_ptr<MultiGPUYolov8TRTInfer> ptr;
  typedef std::shared_ptr<
      cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>>
      TRTInstancePtr;

private:
  std::string m_name;
  std::string m_engine_path;
  int m_max_batch_size;
  std::vector<TRTInstancePtr> m_infer_list;
  std::vector<int> m_device_id_list;

public:
  /*
   * @param name 模型实例名称
   * @param engine_path 模型文件路径
   * @param max_batch_size 最大批处理大小
   * @param device_id_list GPU索引列表
   * @param yolo_type yolo类型
   * @param confidence_threshold 置信度阈值
   * @param nms_threshold nms阈值
   * @return
   */
  explicit MultiGPUYolov8TRTInfer(
      const std::string &name, const std::string &engine_path,
      int max_batch_size, std::vector<int> device_id_list, yolo::Type yolo_type,
      float confidence_threshold = 0.25, float nms_threshold = 0.5);

  std::string get_name() const;

  std::shared_future<yolo::BoxArray> commit(const yolo::Image &input) override;

private:
  int get_infer_index();            // 获取推理设备索引
  std::atomic_int m_infer_index{0}; // 当前使用的推理设备索引
};

} // namespace infer

#endif // TRT_YOLOV8_SERVER_MULTIGPUYOLOV8TRTINFER_H
