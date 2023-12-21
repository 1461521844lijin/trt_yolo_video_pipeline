//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_CONFIGDATA_H
#define VIDEOPIPELINE_CONFIGDATA_H

#include "BaseData.h"

namespace Data {

enum ConfigType {
    CONFIGTYPE_UNKNOWN = 0,
    ROI_CONFIG,       // ROI配置
    ALARM_CONFIG,     // 报警配置
    ANALYSIS_CONFIG,  // 分析配置
    CONFIGTYPE_MAX
};

// 用于存储配置数据的类
class ConfigData : public BaseData {
public:
    using ptr = std::shared_ptr<ConfigData>;

    explicit ConfigData(ConfigType config_type) : BaseData(CONFIG), m_config_type(config_type) {}

    ~ConfigData() override = default;

private:
    ConfigType m_config_type;  // 配置类型
public:
    ConfigType get_config_type() const {
        return m_config_type;
    }
};

}  // namespace Data
#endif  // VIDEOPIPELINE_CONFIGDATA_H
