//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_TRTm_engineH
#define VIDEOPIPELINE_TRTm_engineH

#include "EngineContext.h"
#include "TRTEngine.h"
#include <cstring>
#include <numeric>
#include <unordered_map>

namespace TRT {

enum class DType : int { FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3, BOOL = 4, UINT8 = 5 };

#define Assert(op)                                                                                 \
    do {                                                                                           \
        bool cond = !(!(op));                                                                      \
        if (!cond) {                                                                               \
            INFO("Assert failed, " #op);                                                           \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

#define Assertf(op, ...)                                                                           \
    do {                                                                                           \
        bool cond = !(!(op));                                                                      \
        if (!cond) {                                                                               \
            INFO("Assert failed, " #op " : " __VA_ARGS__);                                         \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

class TRTEngine {
public:
    typedef std::shared_ptr<TRTEngine> ptr;

    TRTEngine() = default;

    static std::shared_ptr<TRTEngine> CreateShared(const std::string &file, int device_id = 0);

    virtual ~TRTEngine() = default;

public:
    bool construct(const void *data, size_t size);

    bool load(const std::string &file, int device_id = 0);

    void setup();

    virtual int index(const std::string &name);

    virtual bool forward(const std::vector<void *> &bindings,
                         void                      *stream,
                         void                      *input_consum_event);

    virtual std::vector<int> run_dims(const std::string &name);

    virtual std::vector<int> run_dims(int ibinding);

    virtual std::vector<int> static_dims(const std::string &name);

    virtual std::vector<int> static_dims(int ibinding);

    virtual int num_bindings();

    virtual bool is_input(int ibinding);

    virtual bool set_run_dims(const std::string &name, const std::vector<int> &dims);

    virtual bool set_run_dims(int ibinding, const std::vector<int> &dims);

    virtual int numel(const std::string &name);

    virtual int numel(int ibinding);

    virtual DType dtype(const std::string &name);

    virtual DType dtype(int ibinding);

    virtual bool has_dynamic_dim();

    virtual void print();

    int  get_device_id() const;
    void set_device_id(int device_id);

private:
    int                                  m_device_id = 0;
    std::shared_ptr<EngineContext>       m_context;
    std::unordered_map<std::string, int> binding_name_to_index_;
};

}  // namespace TRT

#endif  // VIDEOPIPELINE_TRTm_engineH
