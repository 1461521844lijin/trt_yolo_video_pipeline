//#pragma once
//#include "KeyfamePostNode.hpp"
//
//
//namespace Node{
//KeyfamePostNode::KeyfamePostNode(const std::string  name,
//                                     std::string        keyframe_url)
//            :  Node(std::move(name), GraphCore::NODE_TYPE::DES_NODE),
//               m_keyframe_url(keyframe_url){};
//
//// keyframe post node
//Data::BaseData::ptr KeyfamePostNode::handle_data(Data::BaseData::ptr data) {
//    // todo
//    // HttpService
//	HttpService service;
//	std::string img_base64;
//	std::string response;
//    //logger = spdlog::get("logger");
//    auto logger = GetLogger();
//    //std::shared_ptr<spdlog::logger> logger = std::make_shared<spdlog::logger>(logger_ins);
//
//    auto image = data->Get<MAT_IMAGE_TYPE>(MAT_IMAGE);
//    auto status =
//        data->Get<DETECTBOX_FUTURE_TYPE>(DETECTBOX_FUTURE).wait_for(std::chrono::milliseconds(120));
//    if (status == std::future_status::timeout) {
//        INFO("ImageDrawNode: %s wait for future timeout\n", getName().c_str());
//        //logger->error("[{0}:{1}] ImageDrawNode timeout {}", __FILENAME__, __LINE__);
//        //logger->error("[{0}:{1}] ImageDrawNode timeout {}", __FILENAME__, __LINE__);
//
//        return nullptr;
//    }
//    auto box_array = data->Get<DETECTBOX_FUTURE_TYPE>(DETECTBOX_FUTURE).get();
//
//	// post
//    time_t mt = time(NULL);
//    img_base64 = service.b64_encode(image);
//    Json::Value js_task;
//    Json::Value js_sub_target1;
//    //js_task["keyframe"] = "data:image/jpeg;base64" + img_base64;
//    js_task["keyframe"] = img_base64;
//    js_task["task_no"] = "task_no_test";
//    js_task["date"] = mt;
//
//	for (int i=0; i<box_array.size(); i++)
//	{
//		Json::Value temp;
//		temp["index"]   = i;
//		temp["width"]   = box_array[i].right  - box_array[i].left;
//		temp["height"]  = box_array[i].bottom - box_array[i].top;
//		temp["left"]    = box_array[i].left + (box_array[i].right - box_array[i].left)/2;
//		temp["top"]     = box_array[i].top  + (box_array[i].bottom - box_array[i].top)/2;
//		temp["type"]    = box_array[i].class_label;
//		js_task["targets"].append(temp);
//	}
//
//    response = service.requesthttp(js_task.toStyledString(), m_keyframe_url);
//    if(response == "ok")
//    {
//        logger.info("Keyfrmae Post success");
//    }
//    else
//    {
//        logger.info("Keyfrmae Post error");
//    }
//
//    //printf(">>>\n%s\n", js_task.toStyledString().c_str());
//    //this_thread::sleep_for(chrono::milliseconds(1000000));
//    usleep(1000000);
//    return data;
//
//}
//}