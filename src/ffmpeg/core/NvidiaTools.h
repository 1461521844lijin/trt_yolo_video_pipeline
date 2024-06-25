//
// Created by lijin on 2023/7/28.
//

#ifndef VIDEO_FUSION_2_0_NVIDIATOOLS_H
#define VIDEO_FUSION_2_0_NVIDIATOOLS_H

#include "utils/File.h"
#include "utils/String.h"
#include <dlfcn.h>
#include <iostream>

namespace FFmpeg {

static bool checkIfSupportedNvidia_l() {
    auto so = dlopen("libnvcuvid.so.1", RTLD_LAZY);
    if (!so) {
        std::cerr << "libnvcuvid.so.1加载失败" << std::endl;
        return false;
    }
    dlclose(so);
    bool find_driver = false;
    toolkit::File::scanDir(
        "/dev",
        [&](const std::string &path, bool is_dir) {
            if (!is_dir && utils::String::start_with(path, "/dev/nvidia")) {
                // 找到nvidia的驱动
                find_driver = true;
                return false;
            }
            return true;
        },
        false);

    if (!find_driver) {
        std::cerr << "英伟达硬件编解码器驱动文件 /dev/nvidia* 不存在";
    }
    return find_driver;
}

static bool checkIfSupportedNvidia() {
    static auto ret = checkIfSupportedNvidia_l();
    return ret;
}

}  // namespace FFmpeg

#endif  // VIDEO_FUSION_2_0_NVIDIATOOLS_H
