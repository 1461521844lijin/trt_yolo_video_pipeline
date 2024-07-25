//
// Created by Lenovo on 24-7-12.
//

#ifndef VIDEOPIPELINE_MIXMEMORY_H
#define VIDEOPIPELINE_MIXMEMORY_H
#include <cuda_runtime.h>
namespace CUDA {

#define CURRENT_DEVICE_ID -1
/*
 * cpu/gpu混合内存
 * */
class MixMemory {
public:
    MixMemory(int device_id = CURRENT_DEVICE_ID);
    MixMemory(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size);
    virtual ~MixMemory();

    /*
     * 申请gpu内存
     * size: 申请内存大小, 单位: 字节
     * 返回值: 申请的内存地址
     * 如果内存已经申请过, 且大小足够, 则直接返回，否则释放原有内存，重新申请
     */
    void *gpu(size_t size);
    void *cpu(size_t size);

    void  release_gpu();
    void  release_cpu();
    void  release_all();

    inline bool owner_gpu() const {
        return owner_gpu_;
    }
    inline bool owner_cpu() const {
        return owner_cpu_;
    }

    inline size_t cpu_size() const {
        return cpu_size_;
    }
    inline size_t gpu_size() const {
        return gpu_size_;
    }
    inline int device_id() const {
        return device_id_;
    }

    inline void *gpu() const {
        return gpu_;
    }

    // Pinned Memory
    inline void *cpu() const {
        return cpu_;
    }

    void reference_data(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size);

private:
    void  *cpu_       = nullptr;    // cpu内存地址
    size_t cpu_size_  = 0;          // cpu内存大小
    bool   owner_cpu_ = true;       // 是否拥有cpu内存

    int    device_id_ = 0;          // 设备id
    void  *gpu_       = nullptr;    // gpu内存地址
    size_t gpu_size_  = 0;          // gpu内存大小
    bool   owner_gpu_ = true;       // 是否拥有gpu内存
};

}  // namespace CUDA

#endif  // VIDEOPIPELINE_MIXMEMORY_H
