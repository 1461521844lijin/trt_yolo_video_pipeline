#include "infer/MultipleInferenceInstances.h"
#include "trt/yolo/YoloDetectPipeline.h"
#include "trt/yolo/YoloDetectionInfer.h"
#include "utils/json.hpp"
#include "utils/logger.hpp"
// #include "spdlog/spdlog.h"
// #include "spdlog/async.h"
// #include "spdlog/sinks/basic_file_sink.h"
// #include "spdlog/sinks/stdout_color_sinks.h"

// #ifdef WIN32
// #define __FILENAME__ (strrchr(__FILE__, '\\') ? (strrchr(__FILE__, '\\') + 1):__FILE__)
// #else
// #define __FILENAME__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1):__FILE__)
// #endif

using namespace std;

// 加载配置文件
string input_url;
string output_url;
string keyframe_url;
string task_no;

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

// auto logger = spdlog::basic_logger_mt<spdlog::synchronous_factory>("async_file_logger", "logs/log.txt");





int main(int argc, char* argv[]) {
	// Load config
	std::ifstream file("./config.json");
    auto logger = GetLogger(); 
	if (!file.is_open())
	{
        logger.error("[{0}:{1}] Failed load config.json", __FILENAME__, __LINE__);
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
        logger.error("[{0}:{1}] Failed to parse the JSON", __FILENAME__, __LINE__);
		return -1;
    }
    input_url = argv[1];
    output_url = argv[2];
    keyframe_url    = argv[3];
    task_no         = argv[4];
    
    logger.info("[{0}:{1}] {2}", __FILENAME__, __LINE__, input_url.c_str());
    logger.info("[{0}:{1}] {2}", __FILENAME__, __LINE__, output_url.c_str());
    logger.info("[{0}:{1}] {2}", __FILENAME__, __LINE__, keyframe_url.c_str());
    logger.info("[{0}:{1}] {2}", __FILENAME__, __LINE__, task_no.c_str());

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
        //logger.error("[{0}:{1}] Trt_instance init failed", __FILENAME__, __LINE__);
        return -1;
    }

    // 创建处理pipeline
    auto pipeline = std::make_shared<pipeline::YoloDetectPipeline>(
        "test_pipeline", input_url, output_url, keyframe_url, trt_instance, output_width, output_height, output_fps);

    // 启动流水线
    logger.info("[{0}:{1}] Pipeline start", __FILENAME__, __LINE__);

    try {
        pipeline->Start();
    }
    catch (const std::exception& e)
    {
        logger.error("[{0}:{1}] Pipeline init failed", __FILENAME__, __LINE__);
    } 
    //while(true);
    getchar();
    logger.info("[{0}:{1}] Programe exit", __FILENAME__, __LINE__);

}