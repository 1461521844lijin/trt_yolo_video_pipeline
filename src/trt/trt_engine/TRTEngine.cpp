//
// Created by lijin on 2023/12/21.
//

#include "TRTEngine.h"
#include <fstream>
#include <sstream>

namespace TRT {

static std::string format_shape(const nvinfer1::Dims &shape) {
    std::stringstream output;
    char              buf[64];
    const char       *fmts[] = {"%d", "x%d"};
    for (int i = 0; i < shape.nbDims; ++i) {
        snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
        output << buf;
    }
    return output.str();
}

static std::vector<uint8_t> load_file(const std::string &file) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char *)&data[0], length);
    }
    in.close();
    return data;
}

bool TRTEngine::construct(const void *data, size_t size) {
    m_context = std::make_shared<EngineContext>();
    if (!m_context->construct(data, size)) {
        return false;
    }
    setup();
    return true;
}
bool TRTEngine::load(const std::string &file, int device_id) {
    m_device_id = device_id;
    cudaSetDevice(device_id);
    auto data = load_file(file);
    if (data.empty()) {
        INFO("An empty file has been loaded. Please confirm your file path: %s", file.c_str());
        return false;
    }
    return this->construct(data.data(), data.size());
}
void TRTEngine::setup() {
    auto engine     = this->m_context->m_engine;
    int  nbBindings = engine->getNbBindings();

    binding_name_to_index_.clear();
    for (int i = 0; i < nbBindings; ++i) {
        const char *bindingName             = engine->getBindingName(i);
        binding_name_to_index_[bindingName] = i;
    }
}

int TRTEngine::index(const std::string &name) {
    auto iter = binding_name_to_index_.find(name);
    Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s",
            name.c_str());
    return iter->second;
}

bool TRTEngine::forward(const std::vector<void *> &bindings,
                        void                      *stream,
                        void                      *input_consum_event) {
    return this->m_context->m_context->enqueueV2((void **)bindings.data(), (cudaStream_t)stream,
                                                 (cudaEvent_t *)input_consum_event);
}

std::vector<int> TRTEngine::run_dims(const std::string &name) {
    return run_dims(index(name));
}

std::vector<int> TRTEngine::run_dims(int ibinding) {
    auto dim = this->m_context->m_context->getBindingDimensions(ibinding);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
}

std::vector<int> TRTEngine::static_dims(const std::string &name) {
    return static_dims(index(name));
}

std::vector<int> TRTEngine::static_dims(int ibinding) {
    auto dim = this->m_context->m_engine->getBindingDimensions(ibinding);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
}

int TRTEngine::num_bindings() {
    return this->m_context->m_engine->getNbBindings();
}

bool TRTEngine::is_input(int ibinding) {
    return this->m_context->m_engine->bindingIsInput(ibinding);
}

bool TRTEngine::set_run_dims(const std::string &name, const std::vector<int64_t> &dims) {
    return this->set_run_dims(index(name), dims);
}

bool TRTEngine::set_run_dims(int ibinding, const std::vector<int64_t> &dims) {
    nvinfer1::Dims d;
    memcpy(d.d, dims.data(), sizeof(int) * dims.size());
    d.nbDims = dims.size();
    return this->m_context->m_context->setBindingDimensions(ibinding, d);
}

int TRTEngine::numel(const std::string &name) {
    return numel(index(name));
}

int TRTEngine::numel(int ibinding) {
    auto dim = this->m_context->m_context->getBindingDimensions(ibinding);
    return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<int>());
}

DType TRTEngine::dtype(const std::string &name) {
    return dtype(index(name));
}

DType TRTEngine::dtype(int ibinding) {
    return (DType)this->m_context->m_engine->getBindingDataType(ibinding);
}

bool TRTEngine::has_dynamic_dim() {
    // check if any input or output bindings have dynamic shapes
    // code from ChatGPT
    int numBindings = this->m_context->m_engine->getNbBindings();
    for (int i = 0; i < numBindings; ++i) {
        nvinfer1::Dims dims = this->m_context->m_engine->getBindingDimensions(i);
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1)
                return true;
        }
    }
    return false;
}

void TRTEngine::print() {
    INFO("Infer %p [%s]", this, has_dynamic_dim() ? "DynamicShape" : "StaticShape");

    int  num_input  = 0;
    int  num_output = 0;
    auto engine     = this->m_context->m_engine;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (engine->bindingIsInput(i))
            num_input++;
        else
            num_output++;
    }

    INFO("Inputs: %d", num_input);
    for (int i = 0; i < num_input; ++i) {
        auto name = engine->getBindingName(i);
        auto dim  = engine->getBindingDimensions(i);
        INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
    }

    INFO("Outputs: %d", num_output);
    for (int i = 0; i < num_output; ++i) {
        auto name = engine->getBindingName(i + num_input);
        auto dim  = engine->getBindingDimensions(i + num_input);
        INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
    }
}
std::shared_ptr<TRTEngine> TRTEngine::CreateShared(const std::string &file, int device_id) {
    auto engine = std::make_shared<TRTEngine>();
    if (!engine->load(file, device_id)) {
        return nullptr;
    }
    return engine;
}
int TRTEngine::get_device_id() const {
    return m_device_id;
}
void TRTEngine::set_device_id(int device_id) {
    m_device_id = device_id;
}

}  // namespace TRT