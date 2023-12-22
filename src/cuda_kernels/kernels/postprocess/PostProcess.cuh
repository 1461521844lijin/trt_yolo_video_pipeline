//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_POSTPROCESS_CUH
#define VIDEOPIPELINE_POSTPROCESS_CUH

#include <driver_types.h>
namespace CUDA {

/**
 * @brief yolov5 decode kernel invoker
 * @param predict                   模型输出tensor
 * @param num_bboxes                目标框数量
 * @param num_classes               目标类别数量
 * @param confidence_threshold      置信度阈值
 * @param invert_affine_matrix      仿射变换矩阵
 * @param parray                    输出目标框数组
 * @param max_objects               最大目标框数量
 * @param stream                    cuda流
 */
void yolov5_decode_kernel_invoker(float       *predict,
                                  int          num_bboxes,
                                  int          num_classes,
                                  float        confidence_threshold,
                                  float       *invert_affine_matrix,
                                  float       *parray,
                                  int          max_objects,
                                  cudaStream_t stream);
/**
 * @brief nms kernel invoker
 * @param parray        目标框数组地址
 * @param nms_threshold nms阈值
 * @param max_objects   最大目标框数量
 * @param stream        cuda流
 */
void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, cudaStream_t stream);

//
/**
 * @brief yolov8 detect后处理解析
 * @param predict               模型输出tensor
 * @param num_bboxes            目标框数量
 * @param num_classes           目标类别数量
 * @param output_cdim           输出通道维度
 * @param confidence_threshold  置信度阈值
 * @param invert_affine_matrix  仿射变换矩阵
 * @param parray                输出目标框数组
 * @param MAX_IMAGE_BOXES       最大目标框数量
 * @param NUM_BOX_ELEMENT       目标框元素数量
 * @param stream                cuda流
 */
void decode_detect_yolov8_kernel_invoker(float       *predict,
                                         int          num_bboxes,
                                         int          num_classes,
                                         int          output_cdim,
                                         float        confidence_threshold,
                                         float       *invert_affine_matrix,
                                         float       *parray,
                                         int          MAX_IMAGE_BOXES,
                                         int          NUM_BOX_ELEMENT,
                                         cudaStream_t stream);

/**
 * @brief yolov8 segment分支后处理
 * @param left              目标框左上角x坐标
 * @param top               目标框左上角y坐标
 * @param mask_weights      分支权重
 * @param mask_predict      分支预测
 * @param mask_width        分支宽度
 * @param mask_height       分支高度
 * @param mask_out          分支输出
 * @param mask_dim          分支维度
 * @param out_width         输出宽度
 * @param out_height        输出高度
 * @param stream            cuda流
 */
void decode_single_mask(float          left,
                        float          top,
                        float         *mask_weights,
                        float         *mask_predict,
                        int            mask_width,
                        int            mask_height,
                        unsigned char *mask_out,
                        int            mask_dim,
                        int            out_width,
                        int            out_height,
                        cudaStream_t   stream);

/**
 * @brief yolov8 pose后处理解析
 * @param predict               模型输出tensor
 * @param num_bboxes            目标框数量
 * @param pose_num              姿态数量
 * @param output_cdim           输出通道维度
 * @param confidence_threshold  置信度阈值
 * @param invert_affine_matrix  仿射变换矩阵
 * @param parray                输出目标框数组
 * @param MAX_IMAGE_BOXES       最大目标框数量
 * @param NUM_BOX_ELEMENT       目标框元素数量
 * @param stream                cuda流
 */
void decode_pose_yolov8_kernel_invoker(float       *predict,
                                       int          num_bboxes,
                                       int          pose_num,
                                       int          output_cdim,
                                       float        confidence_threshold,
                                       float       *invert_affine_matrix,
                                       float       *parray,
                                       int          MAX_IMAGE_BOXES,
                                       int          NUM_BOX_ELEMENT,
                                       cudaStream_t stream);

}  // namespace CUDA

#endif  // VIDEOPIPELINE_POSTPROCESS_CUH
