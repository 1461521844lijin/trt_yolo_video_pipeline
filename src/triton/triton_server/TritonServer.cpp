//
// Created by Lenovo on 24-7-30.
//

#include "TritonServer.h"
#include "utils/TimeTicker.h"
#include "utils/json11.hpp"
#include "utils/logger.h"
#include <algorithm>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <sstream>

#ifdef TRITON_ENABLE_GPU
#    include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton_server {

#define FAIL(MSG)                                                                                  \
    do {                                                                                           \
        ErrorL << "error: " << (MSG) << std::endl;                                                 \
        exit(1);                                                                                   \
    } while (false)

#define FAIL_IF_ERR(X, MSG)                                                                        \
    do {                                                                                           \
        TRITONSERVER_Error *err__ = (X);                                                           \
        if (err__ != nullptr) {                                                                    \
            ErrorL << "error: " << (MSG) << ": " << TRITONSERVER_ErrorCodeString(err__) << " - "   \
                   << TRITONSERVER_ErrorMessage(err__) << std::endl;                               \
            TRITONSERVER_ErrorDelete(err__);                                                       \
            exit(1);                                                                               \
        }                                                                                          \
    } while (false)

#define THROW_IF_ERR(EX_TYPE, X, MSG)                                                              \
    do {                                                                                           \
        TRITONSERVER_Error *err__ = (X);                                                           \
        if (err__ != nullptr) {                                                                    \
            auto ex__ = (EX_TYPE)(std::string("error: ") + (MSG) + ": " +                          \
                                  TRITONSERVER_ErrorCodeString(err__) + " - " +                    \
                                  TRITONSERVER_ErrorMessage(err__));                               \
            TRITONSERVER_ErrorDelete(err__);                                                       \
            throw ex__;                                                                            \
        }                                                                                          \
    } while (false)

#define IGNORE_ERR(X)                                                                              \
    do {                                                                                           \
        TRITONSERVER_Error *err__ = (X);                                                           \
        if (err__ != nullptr) {                                                                    \
            TRITONSERVER_ErrorDelete(err__);                                                       \
        }                                                                                          \
    } while (false)

#ifdef TRITON_ENABLE_GPU
#    define FAIL_IF_CUDA_ERR(X, MSG)                                                               \
        do {                                                                                       \
            cudaError_t err__ = (X);                                                               \
            if (err__ != cudaSuccess) {                                                            \
                ErrorL << "error: " << (MSG) << ": " << cudaGetErrorString(err__) << std::endl;    \
                exit(1);                                                                           \
            }                                                                                      \
        } while (false)
#endif  // TRITON_ENABLE_GPU

#define JSON_PARSE(JSON, STRING, MSG)                                                              \
    json11::Json JSON;                                                                             \
    {                                                                                              \
        std::string err;                                                                           \
        JSON = json11::Json::parse(STRING, err);                                                   \
        if (!err.empty()) {                                                                        \
            FAIL(MSG + err);                                                                       \
        }                                                                                          \
    }

static size_t getTritonDataTypeByteSize(TRITONSERVER_DataType dtype) {
    size_t byte_size = 0;
    switch (dtype) {
        case TRITONSERVER_TYPE_UINT8:
        case TRITONSERVER_TYPE_INT8: byte_size = sizeof(int8_t); break;
        case TRITONSERVER_TYPE_UINT16:
        case TRITONSERVER_TYPE_INT16: byte_size = sizeof(int16_t); break;
        case TRITONSERVER_TYPE_UINT32:
        case TRITONSERVER_TYPE_INT32:
        case TRITONSERVER_TYPE_FP32: byte_size = sizeof(int32_t); break;
        case TRITONSERVER_TYPE_UINT64:
        case TRITONSERVER_TYPE_INT64:
        case TRITONSERVER_TYPE_FP64: byte_size = sizeof(int64_t); break;
        default:
            FAIL("get invalid datatype " + std::to_string(int(dtype)) +
                 " when get datatype bytesize");
    }
    return byte_size;
}

static TRITONSERVER_DataType convertStrToTritonDataType(std::string datatype_str) {
    if (0 == strcmp(datatype_str.c_str(), "UINT8"))
        return TRITONSERVER_TYPE_UINT8;
    else if (0 == strcmp(datatype_str.c_str(), "UINT16"))
        return TRITONSERVER_TYPE_UINT16;
    else if (0 == strcmp(datatype_str.c_str(), "UINT32"))
        return TRITONSERVER_TYPE_UINT32;
    else if (0 == strcmp(datatype_str.c_str(), "UINT64"))
        return TRITONSERVER_TYPE_UINT64;
    else if (0 == strcmp(datatype_str.c_str(), "INT8"))
        return TRITONSERVER_TYPE_INT8;
    else if (0 == strcmp(datatype_str.c_str(), "INT16"))
        return TRITONSERVER_TYPE_INT16;
    else if (0 == strcmp(datatype_str.c_str(), "INT32"))
        return TRITONSERVER_TYPE_INT32;
    else if (0 == strcmp(datatype_str.c_str(), "INT64"))
        return TRITONSERVER_TYPE_INT64;
    else if (0 == strcmp(datatype_str.c_str(), "FP32"))
        return TRITONSERVER_TYPE_FP32;
    else if (0 == strcmp(datatype_str.c_str(), "FP64"))
        return TRITONSERVER_TYPE_FP64;
    else
        return TRITONSERVER_TYPE_INVALID;
}

static TRITONSERVER_Error *parseModelInfo(const json11::Json &model_metadata,
                                          const json11::Json &model_config,
                                          ModelInfo          &model_info) {
    model_info.platform = model_metadata["platform"].string_value();
    // check model name
    if (model_info.name != model_metadata["name"].string_value()) {
        FAIL("unable to find metadata for model " + model_info.model_key);
    }
    // check model version
    bool found_version = false;
    if (!model_metadata["versions"].is_null()) {
        for (const auto &version : model_metadata["versions"].array_items()) {
            if (version.string_value() == model_info.version) {
                found_version = true;
                break;
            }
        }
    }
    if (!found_version) {
        FAIL("unable to find version " + model_info.version + " status for model " +
             model_info.model_key);
    }

    std::string model_name    = model_info.name;
    std::string model_version = model_info.version;
    std::string model_key     = model_name + ":" + model_version;
    model_info.inputs_dims.clear();
    model_info.outputs_dims.clear();
    for (const auto &input : model_metadata["inputs"].array_items()) {
        std::string           name         = input["name"].string_value();
        std::string           datatype_str = input["datatype"].string_value();
        TRITONSERVER_DataType datatype     = convertStrToTritonDataType(datatype_str);
        if (TRITONSERVER_TYPE_INVALID == datatype) {
            FAIL("model " + model_key + " input:" + name + " contain unsupported datatype " +
                 datatype_str);
        }
        std::vector<int64_t> shape_vec;
        for (const auto &shape_item : input["shape"].array_items()) {
            int64_t dim_value = shape_item.int_value();
            shape_vec.push_back(dim_value);
        }
        model_info.inputs_datatype[name] = datatype;
        model_info.inputs_dims[name]     = shape_vec;
    }
    model_info.max_batch_size = model_config["max_batch_size"].int_value();

    for (const auto &output : model_metadata["outputs"].array_items()) {
        std::string           name         = output["name"].string_value();
        std::string           datatype_str = output["datatype"].string_value();
        TRITONSERVER_DataType datatype     = convertStrToTritonDataType(datatype_str);
        if (TRITONSERVER_TYPE_INVALID == datatype) {
            FAIL("model " + model_key + " output:" + name + " contain unsupported datatype " +
                 datatype_str);
        }
        std::vector<int64_t> shape_vec;
        for (const auto &shape_item : output["shape"].array_items()) {
            int64_t dim_value = shape_item.int_value();
            shape_vec.push_back(dim_value);
        }
        model_info.outputs_datatype[name] = datatype;
        model_info.outputs_dims[name]     = shape_vec;
    }

    return nullptr;
}

static TRITONSERVER_Error *ResponseAlloc(TRITONSERVER_ResponseAllocator *allocator,
                                         const char                     *tensor_name,
                                         size_t                          byte_size,
                                         TRITONSERVER_MemoryType         preferred_memory_type,
                                         int64_t                         preferred_memory_type_id,
                                         void                           *userp,
                                         void                          **buffer,
                                         void                          **buffer_userp,
                                         TRITONSERVER_MemoryType        *actual_memory_type,
                                         int64_t                        *actual_memory_type_id) {
    // Initially attempt to make the actual memory type and id that we
    // allocate be the same as preferred memory type
    *actual_memory_type    = preferred_memory_type;
    *actual_memory_type_id = preferred_memory_type_id;

    // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
    // need to do any other book-keeping.
    if (byte_size == 0) {
        *buffer       = nullptr;
        *buffer_userp = nullptr;
        //        PrintD("allocated " + std::to_string(byte_size) + " bytes for result tensor " +
        //                 tensor_name);
    } else {
        void *allocated_ptr = nullptr;

        switch (*actual_memory_type) {
#ifdef TRITON_ENABLE_GPU
            case TRITONSERVER_MEMORY_CPU_PINNED: {
                auto err = cudaSetDevice(*actual_memory_type_id);
                if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
                    (err != cudaErrorInsufficientDriver)) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string("unable to recover current CUDA device: " +
                                    std::string(cudaGetErrorString(err)))
                            .c_str());
                }

                err = cudaHostAlloc(&allocated_ptr, byte_size, cudaHostAllocPortable);
                if (err != cudaSuccess) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string("cudaHostAlloc failed: " + std::string(cudaGetErrorString(err)))
                            .c_str());
                }
                break;
            }

            case TRITONSERVER_MEMORY_GPU: {
                auto err = cudaSetDevice(*actual_memory_type_id);
                if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
                    (err != cudaErrorInsufficientDriver)) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string("unable to recover current CUDA device: " +
                                    std::string(cudaGetErrorString(err)))
                            .c_str());
                }

                err = cudaMalloc(&allocated_ptr, byte_size);
                if (err != cudaSuccess) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string("cudaMalloc failed: " + std::string(cudaGetErrorString(err)))
                            .c_str());
                }
                break;
            }
#endif  // TRITON_ENABLE_GPU

            // Use CPU memory if the requested memory type is unknown
            // (default case).
            case TRITONSERVER_MEMORY_CPU:
            default: {
                *actual_memory_type = TRITONSERVER_MEMORY_CPU;
                allocated_ptr       = malloc(byte_size);
                break;
            }
        }

        // Pass the tensor name with buffer_userp so we can show it when
        // releasing the buffer.
        if (allocated_ptr != nullptr) {
            *buffer       = allocated_ptr;
            *buffer_userp = new std::string(tensor_name);
            //            VP_DEBUG("allocated " + std::to_string(byte_size) + " bytes in " +
            //                     TRITONSERVER_MemoryTypeString(*actual_memory_type) + " for result
            //                     tensor " + tensor_name);
        }
    }

    return nullptr;  // Success
}

static TRITONSERVER_Error *ResponseRelease(TRITONSERVER_ResponseAllocator *allocator,
                                           void                           *buffer,
                                           void                           *buffer_userp,
                                           size_t                          byte_size,
                                           TRITONSERVER_MemoryType         memory_type,
                                           int64_t                         memory_type_id) {
    std::string *name = nullptr;
    if (buffer_userp != nullptr) {
        name = reinterpret_cast<std::string *>(buffer_userp);
    } else {
        name = new std::string("<unknown>");
    }

    //    VP_DEBUG("Releasing response buffer of size " + std::to_string(byte_size) + +" in " +
    //             TRITONSERVER_MemoryTypeString(memory_type) + +" for result " + *name);

    switch (memory_type) {
        case TRITONSERVER_MEMORY_CPU: free(buffer); break;
#ifdef TRITON_ENABLE_GPU
        case TRITONSERVER_MEMORY_CPU_PINNED: {
            auto err = cudaSetDevice(memory_type_id);
            if (err == cudaSuccess) {
                err = cudaFreeHost(buffer);
            }
            if (err != cudaSuccess) {
                std::cerr << "error: failed to cudaFree " << buffer << ": "
                          << cudaGetErrorString(err) << std::endl;
            }
            break;
        }
        case TRITONSERVER_MEMORY_GPU: {
            auto err = cudaSetDevice(memory_type_id);
            if (err == cudaSuccess) {
                err = cudaFree(buffer);
            }
            if (err != cudaSuccess) {
                std::cerr << "error: failed to cudaFree " << buffer << ": "
                          << cudaGetErrorString(err) << std::endl;
            }
            break;
        }
#endif  // TRITON_ENABLE_GPU
        default:
            std::cerr << "error: unexpected buffer allocated in CUDA managed memory" << std::endl;
            break;
    }
    delete name;
    return nullptr;  // Success
}

static void InferRequestRelease(TRITONSERVER_InferenceRequest *request,
                                const uint32_t                 flags,
                                void                          *userp) {
    // TRITONSERVER_InferenceRequestDelete(request);
    std::promise<void> *barrier = reinterpret_cast<std::promise<void> *>(userp);
    barrier->set_value();
}

static void InferResponseComplete(TRITONSERVER_InferenceResponse *response,
                                  const uint32_t                  flags,
                                  void                           *userp) {
    if (response != nullptr) {
        // Send 'response' to the future.
        std::promise<TRITONSERVER_InferenceResponse *> *p =
            reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse *> *>(userp);
        p->set_value(response);
        delete p;
    }
}

TritonServer &TritonServer::Instance() {
    static TritonServer triton_server;
    return triton_server;
}

void TritonServer::uninit() {
    m_server.reset();
}

void TritonServer::init(const std::string &model_repository_path,
                        int                verbose_level,
                        const std::string &backends_path,
                        const std::string &repo_agent_path,
                        int                timeout) {
    FAIL_IF_ERR(TRITONSERVER_ApiVersion(&m_api_version_major, &m_api_version_minor),
                "getting Triton API version");
    if ((TRITONSERVER_API_VERSION_MAJOR != m_api_version_major) ||
        (TRITONSERVER_API_VERSION_MINOR > m_api_version_minor)) {
        FAIL("triton server API version mismatch");
    }

    m_model_repository_path                    = model_repository_path;
    m_verbose_level                            = verbose_level;
    TRITONSERVER_ServerOptions *server_options = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options), "creating server options");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options,
                                                                 model_repository_path.c_str()),
                "setting model repository path");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
                "setting verbose logging level");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(server_options,
                                                              backends_path.c_str()),
                "setting backend directory");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(server_options,
                                                                repo_agent_path.c_str()),
                "setting repository agent directory");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
                "setting strict model configuration");

#ifdef TRITON_ENABLE_GPU
    double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
#else
    double min_compute_capability = 0;
#endif  // TRITON_ENABLE_GPU
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(server_options,
                                                                           min_compute_capability),
                "setting minimum supported CUDA compute capability");

    // Create the server object using the option settings. The server
    // object encapsulates all the functionality of the Triton server
    // and allows access to the Triton server API. Typically only a
    // single server object is needed by an application, but it is
    // allowed to create multiple server objects within a single
    // application. After the server object is created the server
    // options can be deleted.
    TRITONSERVER_Server *server_ptr = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerNew(&server_ptr, server_options), "creating server object");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsDelete(server_options), "deleting server options");
    std::shared_ptr<TRITONSERVER_Server> server(server_ptr, TRITONSERVER_ServerDelete);
    m_server = std::move(server);
    // Wait until the server is both live and ready. The server will not
    // appear "ready" until all models are loaded and ready to receive
    // inference requests.
    size_t health_iters = 0;
    while (true) {
        bool live = false, ready = false;
        FAIL_IF_ERR(TRITONSERVER_ServerIsLive(m_server.get(), &live),
                    "unable to get server liveness");
        FAIL_IF_ERR(TRITONSERVER_ServerIsReady(m_server.get(), &ready),
                    "unable to get server readiness");
        PrintD("Server Health: live %d, ready %d", live, ready);
        if (live && ready) {
            break;
        }
        if (++health_iters >= 10) {
            FAIL("failed to find healthy inference server");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
    }

    // 输出server metadata
    {
        TRITONSERVER_Message *server_metadata_message;
        FAIL_IF_ERR(TRITONSERVER_ServerMetadata(m_server.get(), &server_metadata_message),
                    "unable to get server metadata message");

        const char *buffer;
        size_t      byte_size;
        FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(server_metadata_message, &buffer,
                                                        &byte_size),
                    "unable to serialize server metadata message");

        PrintD("Triton Server Metadata:");
        PrintD(std::string(buffer, byte_size).c_str());
        FAIL_IF_ERR(TRITONSERVER_MessageDelete(server_metadata_message),
                    "deleting server metadata message");
    }

    // 初始化所有模型信息
    {
        PrintD("init all models info");
        // get model statistic message
        TRITONSERVER_Message *models_statistic_message;
        FAIL_IF_ERR(TRITONSERVER_ServerModelStatistics(m_server.get(), "", -1,
                                                       &models_statistic_message),
                    "unable to get models statistic message");

        const char *buffer;
        size_t      byte_size;
        FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(models_statistic_message, &buffer,
                                                        &byte_size),
                    "unable to serialize models statistic message");

        PrintD(std::string("Triton Server Models Statistics:").c_str());
        std::string models_statistic_str(buffer, byte_size);
        PrintD(models_statistic_str.c_str());
        JSON_PARSE(models_statistic_json, models_statistic_str,
                   "parsing models statistic from JSON");
        FAIL_IF_ERR(TRITONSERVER_MessageDelete(models_statistic_message),
                    "deleting models statistic message");
        // init models info
        auto &model_stats = models_statistic_json["model_stats"].array_items();
        for (auto &model_item : model_stats) {
            ModelInfo model_info;
            model_info.name       = model_item["name"].string_value();
            model_info.version    = model_item["version"].string_value();
            std::string model_key = model_info.name + ":" + model_info.version;
            model_info.model_key  = model_key;

            auto model_metadata_str = get_model_metadata(model_info.name, model_info.version);
            DebugL << model_metadata_str;
            JSON_PARSE(model_metadata_json, model_metadata_str,
                       "parsing model " + model_key + " metadata from JSON");
            auto model_config_str = get_model_config(model_info.name, model_info.version);
            DebugL << model_config_str;
            JSON_PARSE(model_config_json, model_config_str,
                       "parsing model " + model_key + " config from JSON");

            FAIL_IF_ERR(parseModelInfo(model_metadata_json, model_config_json, model_info),
                        "parsing model " + model_key + " metadata");
            if (m_models_info.end() != m_models_info.find(model_key)) {
                WarnL << "arleady init model info: " + model_key;
                continue;
            }
            m_models_info[model_key] = model_info;
        }
    }
}
TritonServer::~TritonServer() {
    uninit();
}
std::string TritonServer::get_model_config(const std::string &model_name,
                                           const std::string &model_version) {
    int64_t               version = std::stoi(model_version);
    std::string           model_config;
    TRITONSERVER_Message *model_config_message;
    FAIL_IF_ERR(TRITONSERVER_ServerModelConfig(m_server.get(), model_name.c_str(), version, 1,
                                               &model_config_message),
                "unable to get model config message");
    const char *buffer;
    size_t      byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_config_message, &buffer, &byte_size),
                "unable to serialize model config message");
    model_config = std::string(buffer, byte_size);
    FAIL_IF_ERR(TRITONSERVER_MessageDelete(model_config_message), "deleting model config message");
    return model_config;
}

std::string TritonServer::get_model_metadata(const std::string &model_name,
                                             const std::string &model_version) {
    int64_t               version = std::stoi(model_version);
    std::string           model_metadata;
    TRITONSERVER_Message *model_metadata_message;
    FAIL_IF_ERR(TRITONSERVER_ServerModelMetadata(m_server.get(), model_name.c_str(), version,
                                                 &model_metadata_message),
                "unable to get model metadata message");
    const char *buffer;
    size_t      byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_metadata_message, &buffer, &byte_size),
                "unable to serialize model metadata message");
    model_metadata = std::string(buffer, byte_size);
    FAIL_IF_ERR(TRITONSERVER_MessageDelete(model_metadata_message),
                "deleting model metadata message");
    return model_metadata;
}

std::string TritonServer::get_model_platform(const std::string &model_name,
                                             const std::string &model_version) {
    // check model exists
    std::string model_key = model_name + ":" + model_version;
    if (m_models_info.end() == m_models_info.find(model_key)) {
        FAIL("cannot not find model info for " + model_key);
    }
    const ModelInfo &model_info = m_models_info[model_key];
    return model_info.platform;
}
void TritonServer::infer(const std::string                   &model_name,
                         const std::string                   &model_version,
                         const std::shared_ptr<TritonTensor> &input_tensor,
                         std::shared_ptr<TritonTensor>       &output_tensor) {
    TimeTicker();
    std::string model_key = model_name + ":" + model_version;
    if (m_models_info.end() == m_models_info.find(model_key)) {
        FAIL("cannot not find model info for " + model_key);
    }
    ModelInfo &model_info = m_models_info[model_key];

    // 1、 创建请求
    TRITONSERVER_ResponseAllocator *allocator = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&allocator, ResponseAlloc, ResponseRelease,
                                                  nullptr /* start_fn */),
                "creating response allocator for model " + model_key);
    TRITONSERVER_InferenceRequest *request = nullptr;
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(&request, m_server.get(), model_name.c_str(),
                                                 std::stoi(model_version)),
                "creating inference request for model " + model_key);
    std::unique_ptr<std::promise<void>> barrier = std::make_unique<std::promise<void>>();
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(
                    request, InferRequestRelease, reinterpret_cast<void *>(barrier.get())),
                "setting request release callback for model " + model_key);
    std::future<void> request_release_future = barrier->get_future();

    // 2、设置输入
    for (auto &[input_name, input_shape] : model_info.inputs_dims) {
        auto   tensor       = input_tensor->tensors[input_name];
        auto   tensor_shape = tensor->dims();
        size_t input_size   = tensor->bytes();
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(request, input_name.c_str(),
                                                          model_info.inputs_datatype[input_name],
                                                          tensor_shape.data(), tensor_shape.size()),
                    "setting input " + input_name + " for model " + model_key);
        if (tensor->head() == CUDA::DataHead::Host) {
            const void *input_base = tensor->cpu();
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                            request, input_name.c_str(), input_base, input_size,
                            TRITONSERVER_MEMORY_CPU_PINNED, 1 /* memory_type_id */),
                        "assigning input: " + input_name + " data for request for " + model_key);
        } else if (tensor->head() == CUDA::DataHead::Device) {
            const void *input_base = tensor->gpu();
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                            request, input_name.c_str(), input_base, input_size,
                            TRITONSERVER_MEMORY_GPU, 1 /* memory_type_id */),
                        "assigning input: " + input_name + " data for request for " + model_key);
        } else {
            FAIL("input tensor " + input_name + " head is invalid for model " + model_key);
        }
    }
    // 设置输出
    for (auto &[output_name, output_shape] : model_info.outputs_dims) {
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(request, output_name.c_str()),
                    "requesting output " + output_name + " for model " + model_key);
    }

    // 3、执行请求
    {
        auto p = new std::promise<TRITONSERVER_InferenceResponse *>();
        std::future<TRITONSERVER_InferenceResponse *> completed = p->get_future();
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                        request, allocator, nullptr /* response_allocator_userp */,
                        InferResponseComplete, reinterpret_cast<void *>(p)),
                    "setting response callback for model " + model_key);
        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_server.get(), request, nullptr /* trace */),
                    "running inference for model " + model_key);
        TRITONSERVER_InferenceResponse *completed_response = completed.get();
        FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(completed_response),
                    "response status for model " + model_key);

        // 4、解析输出
        parseModelInferResponse(completed_response, model_info, output_tensor);

        // delete model infer response
        FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(completed_response),
                    "deleting inference response for model " + model_key);
    }

    // 5、释放资源
    request_release_future.get();
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestDelete(request),
                "deleting inference request for model " + model_key);

    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(allocator),
                "deleting response allocator for model " + model_key);
    return;
}

void TritonServer::parseModelInferResponse(TRITONSERVER_InferenceResponse *response,
                                           ModelInfo                      &model_info,
                                           std::shared_ptr<TritonTensor>  &output_tensor) {
    std::string model_name    = model_info.name;
    std::string model_version = model_info.version;
    std::string model_key     = model_name + ":" + model_version;
    // get model output count
    uint32_t output_count;
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutputCount(response, &output_count),
                "getting number of response outputs for model " + model_key);
    if (output_count != model_info.outputs_dims.size()) {
        FAIL("expecting " + std::to_string(model_info.outputs_dims.size()) +
             " response outputs, got " + std::to_string(output_count) + " for model " + model_key);
    }

    for (uint32_t idx = 0; idx < output_count; ++idx) {
        const char             *cname;
        TRITONSERVER_DataType   datatype;
        const int64_t          *shape;
        uint64_t                dim_count;
        const void             *base;
        size_t                  byte_size;
        TRITONSERVER_MemoryType memory_type;
        int64_t                 memory_type_id;
        void                   *userp;

        FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutput(response, idx, &cname, &datatype, &shape,
                                                         &dim_count, &base, &byte_size,
                                                         &memory_type, &memory_type_id, &userp),
                    "getting output info");

        if (cname == nullptr) {
            FAIL("unable to get output name for model " + model_key);
        }
        std::string name(cname);
        if (model_info.outputs_dims.find(name) == model_info.outputs_dims.end()) {
            FAIL("unexpected output '" + name + "' for model " + model_key);
        }
        auto expected_datatype = model_info.outputs_datatype[name];
        if (datatype != expected_datatype) {
            FAIL("unexpected datatype '" + std::string(TRITONSERVER_DataTypeString(datatype)) +
                 "' for '" + name + "' , model " + model_key);
        }

        // parepare output tensor
        std::vector<int64_t> tensor_shape(shape, shape + dim_count);
        auto                &tensor = output_tensor->tensors[name];
        if (!tensor) {
            FAIL("malloc tensor fail for output " + name + " ,model " + model_key);
        }
        tensor->resize(tensor_shape);
        switch (memory_type) {
            case TRITONSERVER_MEMORY_CPU: {
                DebugL << name + " is stored in system memory for model " + model_key;
                memcpy(tensor->cpu(), base, byte_size);
                break;
            }

            case TRITONSERVER_MEMORY_CPU_PINNED: {
                DebugL << name + " is stored in pinned memory for model " + model_key;
                memcpy(tensor->cpu(), base, byte_size);
                break;
            }

#ifdef TRITON_ENABLE_GPU
            case TRITONSERVER_MEMORY_GPU: {
                DebugL << (name + " is stored in GPU memory for model " + model_key);
                FAIL_IF_CUDA_ERR(cudaMemcpy(tensor->gpu(), base, byte_size, cudaMemcpyDeviceToHost),
                                 "getting " + name + " data from GPU memory for model " +
                                     model_key);
                break;
            }
#endif

            default: FAIL("unexpected memory type for model " + model_key);
        }
    }
}

// triton server 的接口函数太长了，加上每一步增加的宏定义校验，导致整个代码看起来又臭又长
}  // namespace triton_server