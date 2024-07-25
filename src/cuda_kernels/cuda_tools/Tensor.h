
//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_TENSOR_H
#define VIDEOPIPELINE_TENSOR_H

#include "MixMemory.h"
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct CUstream_st;
typedef CUstream_st CUStreamRaw;

namespace CUDA {

typedef struct {
    unsigned short _;
} float16;
typedef CUStreamRaw *CUStream;

enum class DataHead : int {
    Init   = 0,  // 未初始化
    Device = 1,  // 在设备上
    Host   = 2   // 在主机上
};

enum class DataType : int {
    Unknow = -1,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FP16,
    FP32,
    FP64
};
static std::string type_to_string(DataType type) {
    switch (type) {
        case DataType::INT8: return "INT8";
        case DataType::INT16: return "INT16";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
        case DataType::UINT8: return "UINT8";
        case DataType::UINT16: return "UINT16";
        case DataType::UINT32: return "UINT32";
        case DataType::UINT64: return "UINT64";
        case DataType::FP16: return "FP16";
        case DataType::FP32: return "FP32";
        case DataType::FP64: return "FP64";
        default: return "INVALID";
    }
}

static DataType string_to_type(const std::string &type) {
    if (type == "INT8") {
        return DataType::INT8;
    } else if (type == "INT16") {
        return DataType::INT16;
    } else if (type == "INT32") {
        return DataType::INT32;
    } else if (type == "INT64") {
        return DataType::INT64;
    } else if (type == "UINT8") {
        return DataType::UINT8;
    } else if (type == "UINT16") {
        return DataType::UINT16;
    } else if (type == "UINT32") {
        return DataType::UINT32;
    } else if (type == "UINT64") {
        return DataType::UINT64;
    } else if (type == "FP16") {
        return DataType::FP16;
    } else if (type == "FP32") {
        return DataType::FP32;
    } else if (type == "FP64") {
        return DataType::FP64;
    } else {
        throw std::runtime_error("Invalid data type: " + type);
    }
}

static int data_type_size(DataType dt) {
    switch (dt) {
        case DataType::INT8: return 1;
        case DataType::INT16: return 2;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
        case DataType::UINT8: return 1;
        case DataType::UINT16: return 2;
        case DataType::UINT32: return 4;
        case DataType::UINT64: return 8;
        case DataType::FP16: return 2;
        case DataType::FP32: return 4;
        case DataType::FP64: return 8;
        default: return 0;
    }
}

static int data_type_size(const std::string &dt) {
    return data_type_size(string_to_type(dt));

}

static int data_nums(std::vector<int64_t> shape) {
    int nums = 1;
    for (auto &s : shape) {
        assert(s>0);
        nums *= s;
    }
    return nums;
}




float       float16_to_float(float16 value);
float16     float_to_float16(float value);
int         data_type_size(DataType dt);
const char *data_head_string(DataHead dh);

class Tensor {
public:
    typedef std::shared_ptr<Tensor> ptr;

public:
    Tensor(const Tensor &other)            = delete;
    Tensor &operator=(const Tensor &other) = delete;

    explicit Tensor(DataType                   dtype     = DataType::FP32,
                    std::shared_ptr<MixMemory> data      = nullptr,
                    int                        device_id = CURRENT_DEVICE_ID);
    explicit Tensor(int                        n,
                    int                        c,
                    int                        h,
                    int                        w,
                    DataType                   dtype     = DataType::FP32,
                    std::shared_ptr<MixMemory> data      = nullptr,
                    int                        device_id = CURRENT_DEVICE_ID);
    explicit Tensor(int64_t                    ndims,
                    const int64_t             *dims,
                    DataType                   dtype     = DataType::FP32,
                    std::shared_ptr<MixMemory> data      = nullptr,
                    int                        device_id = CURRENT_DEVICE_ID);
    explicit Tensor(const std::vector<int64_t> &dims,
                    DataType                    dtype     = DataType::FP32,
                    std::shared_ptr<MixMemory>  data      = nullptr,
                    int                         device_id = CURRENT_DEVICE_ID);
    virtual ~Tensor();

    // 数据元素个数
    int numel() const;

    // 数据维度
    inline int ndims() const {
        return shape_.size();
    }

    // 数据维度数值
    inline int size(int index) const {
        return shape_[index];
    }
    inline int shape(int index) const {
        return shape_[index];
    }

    inline int batch() const {
        return shape_[0];
    }
    inline int channel() const {
        return shape_[1];
    }
    inline int height() const {
        return shape_[2];
    }
    inline int width() const {
        return shape_[3];
    }

    inline DataType type() const {
        return dtype_;
    }

    // 数据shape
    inline const std::vector<int64_t> &dims() const {
        return shape_;
    }

    // 数据步长
    inline const std::vector<size_t> &strides() const {
        return strides_;
    }

    // 数据字节大小
    inline int bytes() const {
        return bytes_;
    }
    inline int bytes(int start_axis) const {
        return count(start_axis) * element_size();
    }
    inline int element_size() const {
        return data_type_size(dtype_);
    }

    // 数据存放位置
    inline DataHead head() const {
        return head_;
    }

    std::shared_ptr<Tensor> clone() const;
    Tensor                 &release();
    Tensor                 &set_to(float value);
    bool                    empty() const;

    template <typename... _Args>
    int offset(int index, _Args... index_args) const {
        const int index_array[] = {index, index_args...};
        return offset_array(sizeof...(index_args) + 1, index_array);
    }

    int offset_array(const std::vector<int> &index) const;
    int offset_array(size_t size, const int *index_array) const;

    template <typename... _Args>
    Tensor &resize(int64_t dim_size, _Args... dim_size_args) {
        const int64_t dim_size_array[] = {dim_size, dim_size_args...};
        return resize(sizeof...(dim_size_args) + 1, dim_size_array);
    }

    Tensor &resize(int64_t ndims, const int64_t *dims);
    Tensor &resize(const std::vector<int64_t> &dims);
    Tensor &resize_single_dim(int idim, int size);
    int     count(int start_axis = 0) const;
    int     device() const {
        return device_id_;
    }

    Tensor &to_gpu(bool copy = true);
    Tensor &to_cpu(bool copy = true);

    Tensor      &to_half();
    Tensor      &to_float();
    inline void *cpu() const {
        ((Tensor *)this)->to_cpu();
        return data_->cpu();
    }
    inline void *gpu() const {
        ((Tensor *)this)->to_gpu();
        return data_->gpu();
    }

    template <typename DType>
    inline const DType *cpu() const {
        return (DType *)cpu();
    }
    template <typename DType>
    inline DType *cpu() {
        return (DType *)cpu();
    }

    template <typename DType, typename... _Args>
    inline DType *cpu(int i, _Args &&...args) {
        return cpu<DType>() + offset(i, args...);
    }

    template <typename DType>
    inline const DType *gpu() const {
        return (DType *)gpu();
    }
    template <typename DType>
    inline DType *gpu() {
        return (DType *)gpu();
    }

    template <typename DType, typename... _Args>
    inline DType *gpu(int i, _Args &&...args) {
        return gpu<DType>() + offset(i, args...);
    }

    template <typename DType, typename... _Args>
    inline DType &at(int i, _Args &&...args) {
        return *(cpu<DType>() + offset(i, args...));
    }

    std::shared_ptr<MixMemory> get_data() const {
        return data_;
    }
    std::shared_ptr<MixMemory> get_workspace() const {
        return workspace_;
    }
    Tensor &set_workspace(std::shared_ptr<MixMemory> workspace) {
        workspace_ = workspace;
        return *this;
    }

    bool is_stream_owner() const {
        return stream_owner_;
    }
    CUStream get_stream() const {
        return stream_;
    }
    Tensor &set_stream(CUStream stream, bool owner = false) {
        stream_       = stream;
        stream_owner_ = owner;
        return *this;
    }

    Tensor &set_mat(int n, const cv::Mat &image);
    Tensor &set_norm_mat(int n, const cv::Mat &image, float mean[3], float std[3]);
    cv::Mat at_mat(int n = 0, int c = 0) {
        return cv::Mat(height(), width(), CV_32F, cpu<float>(n, c));
    }

    Tensor     &synchronize();
    const char *shape_string() const {
        return shape_string_;
    }
    const char *descriptor() const;

    Tensor &copy_from_gpu(size_t      offset,
                          const void *src,
                          size_t      num_element,
                          int         device_id = CURRENT_DEVICE_ID);
    Tensor &copy_from_cpu(size_t offset, const void *src, size_t num_element);

    void reference_data(const std::vector<int64_t> &shape,
                        void                       *cpu_data,
                        size_t                      cpu_size,
                        void                       *gpu_data,
                        size_t                      gpu_size,
                        DataType                    dtype);

    /**

    # 以下代码是python中加载Tensor
    import numpy as np

    def load_tensor(file):

        with open(file, "rb") as f:
            binary_data = f.read()

        magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
        assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."

        dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

        if dtype == 0:
            np_dtype = np.float32
        elif dtype == 1:
            np_dtype = np.float16
        else:
            assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"

        return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)

     **/
    bool save_to_file(const std::string &file) const;
    bool load_from_file(const std::string &file);

private:
    Tensor &compute_shape_string();
    Tensor &adajust_memory_by_update_dims_or_type();
    void    setup_data(std::shared_ptr<MixMemory> data);

private:
    std::vector<int64_t>       shape_;
    std::vector<size_t>        strides_;
    size_t                     bytes_        = 0;
    DataHead                   head_         = DataHead::Init;
    DataType                   dtype_        = DataType::FP32;
    CUStream                   stream_       = nullptr;
    bool                       stream_owner_ = false;
    int                        device_id_    = 0;
    char                       shape_string_[100];
    char                       descriptor_string_[100];
    std::shared_ptr<MixMemory> data_;
    std::shared_ptr<MixMemory> workspace_;
};

}  // namespace CUDA

#endif  // VIDEOPIPELINE_TENSOR_H
