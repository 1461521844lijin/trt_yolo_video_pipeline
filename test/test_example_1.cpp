#include "infer/MultipleInferenceInstances.h"
#include "trt/yolo/YoloDetectPipeline.h"
#include "trt/yolo/YoloDetectionInfer.h"
#include "utils/json.hpp"
#include "spdlog/spdlog.h"
using namespace std;

// 加载配置文件
string input_url;
string output_url;
string keyframe_url;
string model_path;
string label_path;
int output_width;
int output_height;
int output_fps;
int output_bitrate;
int max_batch_size;
float config_threshold;
float nms_threshold;
bool push_keyframe;

int main() {
	// Load config
	std::ifstream file("../config.json");
	if (!file.is_open())
	{
		printf("Failed load config.json\n");
		return 0;
	}

	Json::CharReaderBuilder json_builder;
	Json::Value root;
	JSONCPP_STRING err;
	std::ifstream ifs;

	if (Json::parseFromStream(json_builder, file, &root, &err)) 
	{
		input_url = root["input_url"].asString();
		output_url = root["output_url"].asString();
		keyframe_url = root["keyframe_url"].asString();
		model_path = root["model_path"].asString();
        label_path = root["label_path"].asString();
		max_batch_size = root["batch_size"].asInt();
        config_threshold = root["config_threshold"].asFloat();
        nms_threshold = root["nms_threshold"].asFloat();
        push_keyframe = root["push_keyframe"].asBool();
        output_width = root["output_width"].asInt();
        output_height = root["output_height"].asInt();
        output_fps = root["output_fps"].asInt();
        output_bitrate = root["output_bitrate"].asInt();
	}
    
	else {
		// 解析失败，打印错误信息
		//std::cout << "Failed to parse the JSON: " << err << std::endl;
		printf("Failed to parse the JSON:%s\n", err.c_str());
		return -1;
    }
    printf("label path: %s", label_path.c_str());
    // std::string input_stream_url(argv[1]);
    // std::string output_stream_url(argv[2]);
    // std::string model_path(argv[3]);
    // std::string label_path(argv[4]);
  

    // int         max_batch_size    = 10;    // 最大batch数
    // float       config_threshold  = 0.25;  // 检测阈值
    // float       nms_threshold     = 0.5;   // nms阈值

    // 模型实例数量列表，列表为模型实例数，每个元素代表该模型实例在哪张显卡上的下标
    std::vector<int> device_list{0, 0, 1, 1};
    auto             type = infer::YoloType::V5;  // 模型类型

    // 创建多卡多实例推理对象
    auto trt_instance =
        std::make_shared<infer::MultipleInferenceInstances<infer::YoloDetectionInfer>>(
            "trt_instance", device_list, model_path, label_path, type, config_threshold,
            nms_threshold, max_batch_size);
    if (!trt_instance->init()) {
        std::cout << "init failed" << std::endl;
        return -1;
    }

    // 创建处理pipeline
    auto pipeline = std::make_shared<pipeline::YoloDetectPipeline>(
        "test_pipeline", input_url, output_url, keyframe_url, trt_instance, output_width, output_height, output_fps);

    // 启动流水线
    pipeline->Start();
    printf("Process done\n");

    getchar();
}