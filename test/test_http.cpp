#include <json/json.h>
#include <fstream>
//#include "utils/HttpService.hpp"


using namespace std;
string input_url;
string output_url;
string keyframe_url;
string engine;
string onnx;
string labels;
int drop_frame;
int fps;
int mode;
int max_batch_size;


int loadconfig(){
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
		engine = root["engine"].asString();
		max_batch_size = root["batch_size"].asInt();
		mode = root["batch_size"].asInt();
		onnx = root["onnx"].asString();
		drop_frame = root["drop_frame"].asInt();
		labels = root["labels"].asString();

        printf("%s\n", input_url.c_str());
	}
    
	else {
		// 解析失败，打印错误信息
		//std::cout << "Failed to parse the JSON: " << err << std::endl;
		printf("Failed to parse the JSON:%s\n", err.c_str());
		return 1;
	}
}

int main()
{

	Json::Value js_task;
	Json::Value js_sub_target1;
	js_task["keyframe"] = "keyframe_test";
	js_task["task_no"] = "task_no_test";
	js_task["date"] = "date_tesk";
	for (size_t i = 0; i < 5; i++)
	{
		Json::Value temp;
		temp["index"] = 1;
		temp["width"] = 1;
		temp["height"] = 1;
		temp["left"] = 1;
		temp["top"] = 1;
		temp["type"] = "car";
		js_task["targets"].append(temp);
	}
	
	std::string str_obj = js_task.toStyledString();
	printf("%s", str_obj.c_str());
	return 0;
}