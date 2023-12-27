//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_PREPROCESS_CUH
#define VIDEOPIPELINE_PREPROCESS_CUH

#include <iostream>

namespace CUDA {

enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

enum class ChannelType : int { None = 0, Invert = 1 };

struct Norm {
    float       mean[3]{};
    float       std[3]{};
    float       alpha{}, beta{};
    NormType    type         = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3],
                         const float std[3],
                         float       alpha        = 1 / 255.0f,
                         ChannelType channel_type = ChannelType::None);

    // out = x * alpha + beta
    static Norm alpha_beta(float       alpha,
                           float       beta         = 0,
                           ChannelType channel_type = ChannelType::None);

    // None
    static Norm None();
};

/*!
 * resizeBilinearAndNormalizeInvoker
 * @details 双线性插值缩放图像并归一化 [0, 255] -> [0, 1]
 * @param src: 输入图像
 * @param src_line_size: 输入图像一行的字节数
 * @param src_width: 输入图像宽度
 * @param src_height: 输入图像高度
 * @param dst: 输出图像
 * @param dst_width: 输出图像宽度
 * @param dst_height: 输出图像高度
 * @param norm: 归一化参数
 * @param stream: cuda流
 */
void resizeBilinearAndNormalizeInvoker(uint8_t     *src,
                                       int          src_line_size,
                                       int          src_width,
                                       int          src_height,
                                       float       *dst,
                                       int          dst_width,
                                       int          dst_height,
                                       const Norm  &norm,
                                       cudaStream_t stream);
/*!
 * warpAffineBilinearAndNormalize Plane
 * @details 仿射变换双线性插值缩放图像并归一化 [0, 255] -> [0, 1]
 * @param src 输入图像
 * @param src_line_size 输入图像一行的字节数
 * @param src_width 输入图像宽度
 * @param src_height 输入图像高度
 * @param dst 输出图像
 * @param dst_width 输出图像宽度
 * @param dst_height 输出图像高度
 * @param matrix_2_3 2x3变换矩阵
 * @param const_value 常量值
 * @param norm 归一化参数
 * @param stream cuda流
 */
void warpAffineBilinearAndNormalizePlaneInvoker(uint8_t     *src,
                                                int          src_line_size,
                                                int          src_width,
                                                int          src_height,
                                                float       *dst,
                                                int          dst_width,
                                                int          dst_height,
                                                float       *matrix_2_3,
                                                uint8_t      const_value,
                                                const Norm  &norm,
                                                cudaStream_t stream);
/*!
 * warpAffineBilinearAndNormalize Focus
 * @details 仿射变换双线性插值缩放图像并归一化 [0, 255] -> [0, 1]
 * @param src 输入图像
 * @param src_line_size 输入图像一行的字节数
 * @param src_width 输入图像宽度
 * @param src_height 输入图像高度
 * @param dst 输出图像
 * @param dst_width 输出图像宽度
 * @param dst_height 输出图像高度
 * @param matrix_2_3 2x3变换矩阵
 * @param const_value 常量值
 * @param norm 归一化参数
 * @param stream cuda流
 */
void warpAffineBilinearAndNormalizeFocusInvoker(uint8_t     *src,
                                                int          src_line_size,
                                                int          src_width,
                                                int          src_height,
                                                float       *dst,
                                                int          dst_width,
                                                int          dst_height,
                                                float       *matrix_2_3,
                                                uint8_t      const_value,
                                                const Norm  &norm,
                                                cudaStream_t stream);

/*!
 * warpPerspective
 * @details 透视变换
 * @param src 输入图像
 * @param src_line_size 输入图像一行的字节数
 * @param src_width 输入图像宽度
 * @param src_height 输入图像高度
 * @param dst 输出图像
 * @param dst_width 输出图像宽度
 * @param dst_height 输出图像高度
 * @param matrix_3_3 3x3变换矩阵
 * @param const_value 常量值
 * @param norm 归一化参数
 * @param stream cuda流
 */
void warpPerspectiveInvoker(uint8_t     *src,
                            int          src_line_size,
                            int          src_width,
                            int          src_height,
                            float       *dst,
                            int          dst_width,
                            int          dst_height,
                            float       *matrix_3_3,
                            uint8_t      const_value,
                            const Norm  &norm,
                            cudaStream_t stream);

/*!
 * normFeature
 * @details 归一化特征
 * @param feature_array 特征数组
 * @param num_feature 特征数量
 * @param feature_length 特征长度
 * @param stream cuda流
 */
void normFeatureInvoker(float       *feature_array,
                        int          num_feature,
                        int          feature_length,
                        cudaStream_t stream);

/*!
 * convertNV12ToBgr
 * @param y y分量
 * @param uv uv分量
 * @param width 宽度
 * @param height 高度
 * @param linesize 一行的字节数
 * @param dst 输出图像
 * @param stream cuda流
 */
void convertNV12ToBgrInvoker(const uint8_t *y,
                             const uint8_t *uv,
                             int            width,
                             int            height,
                             int            linesize,
                             uint8_t       *dst,
                             cudaStream_t   stream);

/**
 * @brief 将BGR格式的图像转换为灰度图
 * @param src
 * @param dst
 * @param width
 * @param height
 */
void bgr2grayInvoker(uint8_t *src, float *dst, int width, int height, cudaStream_t stream = 0);

}  // namespace CUDA

#endif  // VIDEOPIPELINE_PREPROCESS_CUH
