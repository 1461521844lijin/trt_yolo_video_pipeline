// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace hal {

extern const uchar popCountTable[256];

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// forward declarations
int normHamming(const uchar* a, int n);
int normHamming(const uchar* a, const uchar* b, int n);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if CV_AVX2
static inline int _mm256_extract_epi32_(__m256i reg, const int i)
{
    CV_DECL_ALIGNED(32) int reg_data[8];
    CV_DbgAssert(0 <= i && i < 8);
    _mm256_store_si256((__m256i*)reg_data, reg);
    return reg_data[i];
}
#endif

int normHamming(const uchar* a, int n)
{
    CV_AVX_GUARD;

    int i = 0;
    int result = 0;

#if CV_SIMD && CV_SIMD_WIDTH > 16
    {
        v_uint64 t = vx_setzero_u64();
        for (; i <= n - v_uint8::nlanes; i += v_uint8::nlanes)
            t += v_popcount(v_reinterpret_as_u64(vx_load(a + i)));
        result = (int)v_reduce_sum(t);
        vx_cleanup();
    }
#endif

#if CV_POPCNT
    {
#  if defined CV_POPCNT_U64
        for(; i <= n - 8; i += 8)
        {
            result += (int)CV_POPCNT_U64(*(uint64*)(a + i));
        }
#  endif
        for(; i <= n - 4; i += 4)
        {
            result += CV_POPCNT_U32(*(uint*)(a + i));
        }
    }
#elif CV_SIMD
    {
        v_uint64x2 t = v_setzero_u64();
        for(; i <= n - v_uint8x16::nlanes; i += v_uint8x16::nlanes)
            t += v_popcount(v_reinterpret_as_u64(v_load(a + i)));
        result += (int)v_reduce_sum(t);
    }
#endif
#if CV_ENABLE_UNROLLED
    for(; i <= n - 4; i += 4)
    {
        result += popCountTable[a[i]] + popCountTable[a[i+1]] +
        popCountTable[a[i+2]] + popCountTable[a[i+3]];
    }
#endif
    for(; i < n; i++)
    {
        result += popCountTable[a[i]];
    }
    return result;
}

int normHamming(const uchar* a, const uchar* b, int n)
{
    CV_AVX_GUARD;

    int i = 0;
    int result = 0;

#if CV_SIMD && CV_SIMD_WIDTH > 16
    {
        v_uint64 t = vx_setzero_u64();
        for (; i <= n - v_uint8::nlanes; i += v_uint8::nlanes)
            t += v_popcount(v_reinterpret_as_u64(vx_load(a + i) ^ vx_load(b + i)));
        result += (int)v_reduce_sum(t);
    }
#endif

#if CV_POPCNT
    {
#  if defined CV_POPCNT_U64
        for(; i <= n - 8; i += 8)
        {
            result += (int)CV_POPCNT_U64(*(uint64*)(a + i) ^ *(uint64*)(b + i));
        }
#  endif
        for(; i <= n - 4; i += 4)
        {
            result += CV_POPCNT_U32(*(uint*)(a + i) ^ *(uint*)(b + i));
        }
    }
#elif CV_SIMD
    {
        v_uint64x2 t = v_setzero_u64();
        for(; i <= n - v_uint8x16::nlanes; i += v_uint8x16::nlanes)
            t += v_popcount(v_reinterpret_as_u64(v_load(a + i) ^ v_load(b + i)));
        result += (int)v_reduce_sum(t);
    }
#endif
#if CV_ENABLE_UNROLLED
    for(; i <= n - 4; i += 4)
    {
        result += popCountTable[a[i] ^ b[i]] + popCountTable[a[i+1] ^ b[i+1]] +
                popCountTable[a[i+2] ^ b[i+2]] + popCountTable[a[i+3] ^ b[i+3]];
    }
#endif
    for(; i < n; i++)
    {
        result += popCountTable[a[i] ^ b[i]];
    }
    return result;
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} //cv::hal
