
#include <opencv2/opencv.hpp>

#include "trt/cpm.hpp"
#include "trt/infer.hpp"
#include "trt/yolo.hpp"

using namespace std;

static const char *cocolabels[] = {"person",        "bicycle",      "car",
                                   "motorcycle",    "airplane",     "bus",
                                   "train",         "truck",        "boat",
                                   "traffic light", "fire hydrant", "stop sign",
                                   "parking meter", "bench",        "bird",
                                   "cat",           "dog",          "horse",
                                   "sheep",         "cow",          "elephant",
                                   "bear",          "zebra",        "giraffe",
                                   "backpack",      "umbrella",     "handbag",
                                   "tie",           "suitcase",     "frisbee",
                                   "skis",          "snowboard",    "sports ball",
                                   "kite",          "baseball bat", "baseball glove",
                                   "skateboard",    "surfboard",    "tennis racket",
                                   "bottle",        "wine glass",   "cup",
                                   "fork",          "knife",        "spoon",
                                   "bowl",          "banana",       "apple",
                                   "sandwich",      "orange",       "broccoli",
                                   "carrot",        "hot dog",      "pizza",
                                   "donut",         "cake",         "chair",
                                   "couch",         "potted plant", "bed",
                                   "dining table",  "toilet",       "tv",
                                   "laptop",        "mouse",        "remote",
                                   "keyboard",      "cell phone",   "microwave",
                                   "oven",          "toaster",      "sink",
                                   "refrigerator",  "book",         "clock",
                                   "vase",          "scissors",     "teddy bear",
                                   "hair drier",    "toothbrush"};

yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

void perf() {
  int max_infer_batch = 16;
  int batch = 16;
  std::vector<cv::Mat> images{cv::imread("/root/trt_projects/infer-main/workspace/inference/car.jpg"), cv::imread("/root/trt_projects/infer-main/workspace/inference/gril.jpg"),
                              cv::imread("/root/trt_projects/infer-main/workspace/inference/group.jpg")};

  for (int i = images.size(); i < batch; ++i) images.push_back(images[i % 3]);

  cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
  bool ok = cpmi.start([] { return yolo::load("/root/trt_projects/infer-main/workspace/yolov8n.transd.engine", yolo::Type::V8); },
                       max_infer_batch);

  if (!ok) return;

  std::vector<yolo::Image> yoloimages(images.size());
  std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);

  trt::Timer timer;
  for (int i = 0; i < 5; ++i) {
    timer.start();
    cpmi.commits(yoloimages).back().get();
    timer.stop("BATCH16");
  }

  for (int i = 0; i < 5; ++i) {
    timer.start();
    cpmi.commit(yoloimages[0]).get();
    timer.stop("BATCH1");
  }
}

void batch_inference() {
  std::vector<cv::Mat> images{cv::imread("/root/trt_projects/infer-main/workspace/inference/car.jpg"),
                              cv::imread("/root/trt_projects/infer-main/workspace/inference/gril.jpg"),
                              cv::imread("/root/trt_projects/infer-main/workspace/inference/group.jpg")};
  auto yolo = yolo::load("/root/trt_projects/infer-main/workspace/yolov8n.transd.fp16.engine", yolo::Type::V8);
  if (yolo == nullptr) return;

  std::vector<yolo::Image> yoloimages(images.size());
  std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);
  auto batched_result = yolo->forwards(yoloimages);
  for (int ib = 0; ib < (int)batched_result.size(); ++ib) {
    auto &objs = batched_result[ib];
    auto &image = images[ib];
    for (auto &obj : objs) {
      uint8_t b, g, r;
      tie(b, g, r) = yolo::random_color(obj.class_label);
      cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);

      auto name = cocolabels[obj.class_label];
      auto caption = cv::format("%s %.2f", name, obj.confidence);
      int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
      cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                    cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
      cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                  16);
    }
    printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
    cv::imwrite(cv::format("/root/trt_projects/infer-main/workspace/result/Result%d.jpg", ib), image);
  }
}

void single_inference() {
  cv::Mat image = cv::imread("/root/trt_projects/infer-main/workspace/inference/car.jpg");
  auto yolo = yolo::load("/root/trt_projects/infer-main/workspace/yolov8n-seg.b1.transd.engine", yolo::Type::V8Seg);
  if (yolo == nullptr) return;

  auto objs = yolo->forward(cvimg(image));
  int i = 0;
  for (auto &obj : objs) {
    uint8_t b, g, r;
    tie(b, g, r) = yolo::random_color(obj.class_label);
    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                  cv::Scalar(b, g, r), 5);

    auto name = cocolabels[obj.class_label];
    auto caption = cv::format("%s %.2f", name, obj.confidence);
    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                  cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
    cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);

    if (obj.seg) {
      cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);
      // 转为三通道图像并赋予随机颜色
      cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
      // 阈值化
      cv::threshold(mask, mask, 50, 255, cv::THRESH_BINARY);
      // 生成随机颜色
      cv::Vec3b color = cv::Vec3b(b, g, r);
      // 将掩码中的255像素赋予随机颜色
      mask.setTo(color, mask > cv::Vec3b(155, 155, 155));
      cv::imwrite(cv::format("%d_mask.jpg", i), mask);
      // 将掩码缩放到原图像大小
      cv::Mat resize_mask;
      cv::resize(mask, resize_mask, cv::Size(obj.right - obj.left, obj.bottom - obj.top));
      // 获取原始图像的ROI
      cv::Mat imageROI = image(cv::Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top));
      // 将掩码叠加到ROI上
      cv::addWeighted(imageROI, 1, resize_mask, 0.8, 1, imageROI);
      cv::imwrite(cv::format("%d_mask_image.jpg", i), image);
      i++;
   }
  }

  printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
  cv::imwrite("/root/trt_projects/infer-main/workspace/result/Result.jpg", image);
}

int main() {
  perf();
  batch_inference();
  single_inference();
  return 0;
}