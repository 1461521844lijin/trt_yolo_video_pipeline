// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2022 Intel Corporation


#ifndef OPENCV_GAPI_GCPUBACKEND_HPP
#define OPENCV_GAPI_GCPUBACKEND_HPP

#include <map>                // map
#include <unordered_map>      // unordered_map
#include <tuple>              // tuple
#include <ade/util/algorithm.hpp> // type_list_index

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include "api/gorigin.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gislandmodel.hpp"

namespace cv { namespace gimpl {

struct CPUUnit
{
    static const char *name() { return "HostKernel"; }
    GCPUKernel k;
};

class GCPUExecutable final: public GIslandExecutable
{
    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;
    cv::GCompileArgs m_compileArgs;

    struct OperationInfo
    {
        ade::NodeHandle nh;
        GMetaArgs expected_out_metas;
    };

    // Execution script, currently absolutely naive
    std::vector<OperationInfo> m_script;

    // TODO: Check that it is thread-safe
    // Map of stateful kernel nodes to their kernels' states
    std::unordered_map<ade::NodeHandle, GArg,
                       ade::HandleHasher<ade::Node>> m_nodesToStates;

    // List of all resources in graph (both internal and external)
    std::vector<ade::NodeHandle> m_dataNodes;
    std::vector<ade::NodeHandle> m_opNodes;

    // Actual data of all resources in graph (both internal and external)
    Mag m_res;

    // A flag for call_once() (used for log warnings)
    std::once_flag m_warnFlag;

    GArg packArg(const GArg &arg);
    void setupKernelStates();

    void makeReshape();

public:
    GCPUExecutable(const ade::Graph                   &graph,
                   const cv::GCompileArgs             &compileArgs,
                   const std::vector<ade::NodeHandle> &nodes);

    virtual inline bool canReshape() const override { return true; }
    virtual void reshape(ade::Graph&, const GCompileArgs&) override;

    virtual void handleNewStream() override;

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;
};

}}

#endif // OPENCV_GAPI_GCPUBACKEND_HPP
