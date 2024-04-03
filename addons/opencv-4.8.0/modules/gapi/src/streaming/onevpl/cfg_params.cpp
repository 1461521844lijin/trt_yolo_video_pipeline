// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <sstream>
#include <opencv2/gapi/util/throw.hpp>

#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

namespace util {
struct variant_comparator : cv::util::static_visitor<bool, variant_comparator> {
    variant_comparator(const CfgParam::value_t& rhs_value) :
        rhs(rhs_value) {}

    template<typename ValueType>
    bool visit(const ValueType& lhs) const {
        return lhs < cv::util::get<ValueType>(rhs);
    }
private:
    const CfgParam::value_t& rhs;
};

struct variant_stringifier : cv::util::static_visitor<std::string, variant_stringifier> {
    template<typename ValueType>
    std::string visit(const ValueType& lhs) const {
        std::stringstream ss;
        ss << lhs;
        return ss.str();
    }
};
} // namespace util

struct CfgParam::Priv {
    Priv(const std::string& param_name, CfgParam::value_t&& param_value, bool is_major_param) :
        name(param_name), value(std::forward<value_t>(param_value)), major_flag(is_major_param) {
    }

    const CfgParam::name_t& get_name_impl() const {
        return name;
    }

    const CfgParam::value_t& get_value_impl() const {
        return value;
    }

    bool is_major_impl() const {
        return major_flag;
    }

    // comparison implementation
    bool operator< (const Priv& rhs) const {
        // implement default pair comparison
        if (get_name_impl() < rhs.get_name_impl()) {
            return true;
        } else if (get_name_impl() > rhs.get_name_impl()) {
            return false;
        }

        //TODO implement operator < for cv::util::variant
        const CfgParam::value_t& lvar = get_value_impl();
        const CfgParam::value_t& rvar = rhs.get_value_impl();
        if (lvar.index() < rvar.index()) {
            return true;
        } else if (lvar.index() > rvar.index()) {
            return false;
        }

        util::variant_comparator comp(rvar);
        return cv::util::visit(comp, lvar);
    }

    bool operator==(const Priv& rhs) const {
        return (get_name_impl() == rhs.get_name_impl())
                && (get_value_impl() == rhs.get_value_impl());
    }

    bool operator!=(const Priv& rhs) const {
        return !(*this == rhs);
    }

    CfgParam::name_t name;
    CfgParam::value_t value;
    bool major_flag;
};

CfgParam::CfgParam (const std::string& param_name, value_t&& param_value, bool is_major_param) :
    m_priv(new Priv(param_name, std::move(param_value), is_major_param)) {
}

CfgParam::~CfgParam() = default;

CfgParam CfgParam::create_frames_pool_size(size_t value) {
    // NB: cast to uint64_t because CfgParam inner variant works over
    // uint64_t instead of size_t and mirrored VPL types variety
    // but size_t looks more friendly for C++ high-level development
    return CfgParam::create(CfgParam::frames_pool_size_name(),
                            static_cast<uint64_t>(value), false);
}

CfgParam CfgParam::create_acceleration_mode(uint32_t value) {
    return CfgParam::create(CfgParam::acceleration_mode_name(), value);
}

CfgParam CfgParam::create_acceleration_mode(const char* value) {
    return CfgParam::create(CfgParam::acceleration_mode_name(), std::string(value));
}

CfgParam CfgParam::create_decoder_id(uint32_t value) {
    return CfgParam::create(CfgParam::decoder_id_name(), value);
}

CfgParam CfgParam::create_decoder_id(const char* value) {
    return CfgParam::create(CfgParam::decoder_id_name(), std::string(value));
}

CfgParam CfgParam::create_implementation(uint32_t value) {
    return CfgParam::create(CfgParam::implementation_name(), value);
}

CfgParam CfgParam::create_implementation(const char* value) {
    return CfgParam::create(CfgParam::implementation_name(), std::string(value));
}

CfgParam CfgParam::create_vpp_frames_pool_size(size_t value) {
    // NB: cast to uint64_t because CfgParam inner variant works over
    // uint64_t instead of size_t and mirrored VPL types variety
    // but size_t looks more friendly for C++ high-level development
    return CfgParam::create(CfgParam::vpp_frames_pool_size_name(),
                            static_cast<uint64_t>(value), false);
}

CfgParam CfgParam::create_vpp_in_width(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_in_width_name(), value, false);
}

CfgParam CfgParam::create_vpp_in_height(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_in_height_name(), value, false);
}

CfgParam CfgParam::create_vpp_in_crop_x(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_in_crop_x_name(), value, false);
}

CfgParam CfgParam::create_vpp_in_crop_y(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_in_crop_y_name(), value, false);
}

CfgParam CfgParam::create_vpp_in_crop_w(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_in_crop_w_name(), value, false);
}

CfgParam CfgParam::create_vpp_in_crop_h(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_in_crop_h_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_fourcc(uint32_t value) {
    return CfgParam::create(CfgParam::vpp_out_fourcc_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_chroma_format(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_out_chroma_format_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_width(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_out_width_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_height(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_out_height_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_crop_x(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_out_crop_x_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_crop_y(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_out_crop_y_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_crop_w(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_out_crop_w_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_crop_h(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_out_crop_h_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_pic_struct(uint16_t value) {
    return CfgParam::create(CfgParam::vpp_out_pic_struct_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_framerate_n(uint32_t value) {
    return CfgParam::create(CfgParam::vpp_out_framerate_n_name(), value, false);
}

CfgParam CfgParam::create_vpp_out_framerate_d(uint32_t value) {
    return CfgParam::create(CfgParam::vpp_out_framerate_d_name(), value, false);
}

CfgParam& CfgParam::operator=(const CfgParam& src) {
    if (this != &src) {
        m_priv = src.m_priv;
    }
    return *this;
}

CfgParam& CfgParam::operator=(CfgParam&& src) {
    if (this != &src) {
        m_priv = std::move(src.m_priv);
    }
    return *this;
}

CfgParam::CfgParam(const CfgParam& src) :
    m_priv(src.m_priv) {
}

CfgParam::CfgParam(CfgParam&& src) :
    m_priv(std::move(src.m_priv)) {
}

const CfgParam::name_t& CfgParam::get_name() const {
    return m_priv->get_name_impl();
}

const CfgParam::value_t& CfgParam::get_value() const {
    return m_priv->get_value_impl();
}

bool CfgParam::is_major() const {
    return m_priv->is_major_impl();
}

std::string CfgParam::to_string() const {
    return get_name() + ":" + cv::util::visit(util::variant_stringifier{},
                                              get_value());
}

bool CfgParam::operator< (const CfgParam& rhs) const {
    return *m_priv < *rhs.m_priv;
}

bool CfgParam::operator==(const CfgParam& rhs) const {
    return *m_priv == *rhs.m_priv;
}

bool CfgParam::operator!=(const CfgParam& rhs) const {
    return *m_priv != *rhs.m_priv;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
