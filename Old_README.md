# trt_yolov8_infer_example

基于手写AI的infer模块，使用ffmpeg拉流解码送入trt_yolov8进行目标检测，渲染后推流输出画面，支持cuda硬件编解码

## 说明

    感谢杜老这么棒的开源代码，调用起来简单又高效，本项目就是基于infer模块添加了ffmpeg的拉流解码编码推流流程
    在项目中设计了一个方便算法异步调用的责任链类，将解码、算法推理、编码都放在了不同的线程中，方便高效的进行多路并发推理和业务处理
    责任链模块也很可以很方便的进行扩展，比如增加一个线程用于数据的前处理或者后处理，或者增加一个线程用于数据的存储等等
    其中trt模块使用的代码原项目地址：
    https://github.com/shouxieai/infer
    如何转换onnx模型为trt或者配置环境请参考原项目

## 思路

    trt infer中的cpm模块本身就是一个独立的消费者线程，而且使用了c++中的future和promise机制，可以方便的进行多线程间的数据流动
    因此只需加载一个全局的推理cpm模块作为推理线程，每当有一路视频时生成一个消费者给cpm模块commit数据，会直接返回拿到一份promise
    将这个promise放入我们指定的异步线程中进行结果等待，用这机制保证多路的数据是在我们指定的线程中运行,且都是在流水线中异步的
    只要trt多batch的推理性能足够好，就可以很方便的进行多路的推理了

## 如何使用

先创建一个trt_infer实例

~~~c++
    std::string model_path = "/root/trt_projects/infer-main/workspace/yolov8n.transd.engine";
        
    std::shared_ptr<cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>> trt_instance;
    trt_instance = std::make_shared<cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>>();
    trt_instance->start([&]{ 
        return yolo::load(model_path_seg, yolo::Type::V8Seg);}
        ,max_batch_size
        );
~~~

创建流水线中的异步执行节点

~~~c++
    // ffmpeg解码节点
    auto ffmpeg_input_node = FFmpeg::create_ffmpeg("ffmpeg_input_node", stream_url);
    // trt推理节点
    auto trt_node = std::make_shared<trt::TrtNode>("trt_node");
    // 结果渲染节点
    auto trt_draw_node = std::make_shared<trt::ImageDrawNode>("trt_draw_node");
    // 推流输出节点
    auto ffmpeg_output_node = std::make_shared<FFmpeg::StreamPusher>("ffmpeg_output_node");
~~~

将trt实例放入trt_node中，这里的trt_node就是充当一个生产者节点和trt实例（消费者）的桥梁，保证我们多路时的数据在指定的线程中运行的

~~~c++
    trt_node->set_trt_instance(trt_instance);
~~~

将各个节点串联起来，并且启动

~~~c++
    Base::LinkNode(ffmpeg_input_node, trt_node);
    Base::LinkNode(trt_node, trt_draw_node);
    Base::LinkNode(trt_draw_node, ffmpeg_output_node);
    
    trt_draw_node->Start();
    ffmpeg_output_node->Start();
    trt_node->Start();
    ffmpeg_input_node->Start();
    // 节点只要在同一时间启动不用关注启动顺序，但极端情况下可以选择从后往前一次启动   
~~~

若多路并发推理，只需创建多个流水线即可
单卡单模型场景下trt_instance只需创建一个即可,infer中的生产者消费者模型已经提供了很好的处理动态多batch下的模型推理性能了

## 性能测试

        ffmpeg中将各种缓存设置为最小值，编码缓冲和预制选项设置最快，单路画面延迟是小于1s的，如果支持webrtc的话延迟可以更低
        A30显卡 1080p视频输入 1080p视频编码推流输出  yolov8n模型加载了4个infer实例 并发28路 gpu利用率90%左右 
        
        代码是支持硬件编解码的，但需要ffmpeg重新编译支持后测试
        要开启硬件解码的话使用 需要指定cmake的编译选项
