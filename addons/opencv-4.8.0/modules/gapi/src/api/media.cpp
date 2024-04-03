// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "precomp.hpp"
#include <opencv2/gapi/media.hpp>

struct cv::MediaFrame::Priv {
    std::unique_ptr<IAdapter> adapter;
};

cv::MediaFrame::MediaFrame() {
}

cv::MediaFrame::MediaFrame(AdapterPtr &&ptr)
    : m(new Priv{std::move(ptr)}) {
}

cv::GFrameDesc cv::MediaFrame::desc() const {
    return m->adapter->meta();
}

cv::MediaFrame::View cv::MediaFrame::access(Access code) const {
    return m->adapter->access(code);
}

cv::util::any cv::MediaFrame::blobParams() const
{
    return m->adapter->blobParams();
}

cv::MediaFrame::IAdapter* cv::MediaFrame::getAdapter() const {
    return m->adapter.get();
}

void cv::MediaFrame::serialize(cv::gapi::s11n::IOStream& os) const {
    m->adapter->serialize(os);
}

cv::MediaFrame::View::View(Ptrs&& ptrs, Strides&& strs, Callback &&cb)
    : ptr   (std::move(ptrs))
    , stride(std::move(strs))
    , m_cb  (std::move(cb)) {
}

cv::MediaFrame::View::~View() {
    if (m_cb) {
        m_cb();
    }
}

cv::util::any cv::MediaFrame::IAdapter::blobParams() const
{
    // Does nothing by default
    return {};
}

cv::MediaFrame::IAdapter::~IAdapter() {
}
