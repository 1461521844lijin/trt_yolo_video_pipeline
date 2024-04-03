// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation

#ifndef OPENCV_GAPI_GSTREAMING_EXECUTOR_HPP
#define OPENCV_GAPI_GSTREAMING_EXECUTOR_HPP

#ifdef _MSC_VER
#pragma warning(disable: 4503)  // "decorated name length exceeded"
                                // on concurrent_bounded_queue
#endif

#include <thread> // thread
#include <vector>
#include <unordered_map>

#if defined(HAVE_TBB)
#  include <tbb/concurrent_queue.h> // FIXME: drop it from here!
template<typename T> using QueueClass = tbb::concurrent_bounded_queue<T>;
#else
#  include "executor/conc_queue.hpp"
template<typename T> using QueueClass = cv::gapi::own::concurrent_bounded_queue<T>;
#endif // TBB
#include "executor/last_value.hpp"

#include "executor/gabstractstreamingexecutor.hpp"

namespace cv {
namespace gimpl {

namespace stream {
struct Start {};
struct Stop {
    enum class Kind {
        HARD, // a hard-stop: end-of-pipeline reached or stop() called
        CNST, // a soft-stop emitted for/by constant sources (see QueueReader)
    } kind = Kind::HARD;
    cv::GRunArg cdata; // const data for CNST stop
};

struct Result {
    cv::GRunArgs      args;  // Full results vector
    std::vector<bool> flags; // Availability flags (in case of desync)
};

using Cmd = cv::util::variant
    < cv::util::monostate
    , Start                // Tells emitters to start working. Not broadcasted to workers.
    , Stop                 // Tells emitters to stop working. Broadcasted to workers.
    , cv::GRunArg          // Workers data payload to process.
    , Result               // Pipeline's data for gout()
    , cv::gimpl::Exception // Exception which is thrown while execution.
   >;

// Interface over a queue. The underlying queue implementation may be
// different. This class is mainly introduced to bring some
// abstraction over the real queues (bounded in-order) and a
// desynchronized data slots (see required to implement
// cv::gapi::desync)

class Q {
public:
    virtual void push(const Cmd &cmd) = 0;
    virtual void pop(Cmd &cmd) = 0;
    virtual bool try_pop(Cmd &cmd) = 0;
    virtual void clear() = 0;
    virtual ~Q() = default;
};

// A regular queue implementation
class SyncQueue final: public Q {
    QueueClass<Cmd> m_q;    // FIXME: OWN or WRAP??

public:
    virtual void push(const Cmd &cmd) override { m_q.push(cmd); }
    virtual void pop(Cmd &cmd)        override { m_q.pop(cmd);  }
    virtual bool try_pop(Cmd &cmd)    override { return m_q.try_pop(cmd); }
    virtual void clear()              override { m_q.clear(); }

    void set_capacity(std::size_t c) { m_q.set_capacity(c);}
};

// Desynchronized "queue" implementation
// Every push overwrites value which is not yet popped
// This container can hold 0 or 1 element
// Special handling for Stop is implemented (FIXME: not really)
class DesyncQueue final: public Q {
    cv::gapi::own::last_written_value<Cmd> m_v;

public:
    virtual void push(const Cmd &cmd) override { m_v.push(cmd); }
    virtual void pop(Cmd &cmd)        override { m_v.pop(cmd);  }
    virtual bool try_pop(Cmd &cmd)    override { return m_v.try_pop(cmd); }
    virtual void clear()              override { m_v.clear(); }
};

} // namespace stream

// FIXME: Currently all GExecutor comments apply also
// to this one. Please document it separately in the future.

class GStreamingExecutor final: public GAbstractStreamingExecutor
{
protected:
    // GStreamingExecutor is a state machine described as follows
    //
    //              setSource() called
    //   STOPPED:  - - - - - - - - - ->READY:
    //   --------                      ------
    //   Initial state                 Input data specified
    //   No threads running            Threads are created and IDLE
    //   ^                             (Currently our emitter threads
    //   :                             are bounded to input data)
    //   : stop() called               No processing happending
    //   : OR                          :
    //   : end-of-stream reached       :  start() called
    //   : during pull()/try_pull()    V
    //   :                             RUNNING:
    //   :                             --------
    //   :                             Actual pipeline execution
    //    - - - - - - - - - - - - - -  Threads are running
    //
    enum class State {
        STOPPED,
        READY,
        RUNNING,
    } state = State::STOPPED;

    cv::GMetaArgs m_last_metas;
    util::optional<bool> m_reshapable;

    // FIXME: Naive executor details are here for now
    // but then it should be moved to another place
    struct OpDesc
    {
        std::vector<RcDesc> in_objects;
        std::vector<RcDesc> out_objects;
        cv::GMetaArgs       out_metas;
        ade::NodeHandle     nh;

        cv::GRunArgs in_constants;

        std::shared_ptr<GIslandExecutable> isl_exec;
    };
    std::vector<OpDesc> m_ops;

    struct DataDesc
    {
        ade::NodeHandle slot_nh;
        ade::NodeHandle data_nh;
    };
    std::vector<DataDesc> m_slots;

    cv::GRunArgs m_const_vals;

    // Order in these vectors follows the GComputaion's protocol
    std::vector<ade::NodeHandle> m_emitters;
    std::vector<ade::NodeHandle> m_sinks;

    class Synchronizer;
    std::unique_ptr<Synchronizer> m_sync;

    std::vector<std::thread> m_threads;
    std::vector<stream::SyncQueue>   m_emitter_queues;

    // a view over m_emitter_queues
    std::vector<stream::SyncQueue*>  m_const_emitter_queues;

    std::vector<stream::Q*>          m_sink_queues;

    // desync path tags for outputs. -1 means that output
    // doesn't belong to a desync path
    std::vector<int>                 m_sink_sync;

    std::unordered_set<stream::Q*>   m_internal_queues;
    stream::SyncQueue m_out_queue;

    // Describes mapping from desync paths to collector threads
    struct CollectorThreadInfo {
        std::vector<stream::Q*>  queues;
        std::vector<int> mapping;
    };
    std::unordered_map<int, CollectorThreadInfo> m_collector_map;


    void wait_shutdown();

    cv::GTypesInfo out_info;

public:
    explicit GStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model,
                                const cv::GCompileArgs &comp_args);
    ~GStreamingExecutor();
    void setSource(GRunArgs &&args) override;
    void start() override;
    bool pull(cv::GRunArgsP &&outs) override;
    bool pull(cv::GOptRunArgsP &&outs) override;
    PyPullResult pull() override;
    bool try_pull(cv::GRunArgsP &&outs) override;
    void stop() override;
    bool running() const override;
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_EXECUTOR_HPP
