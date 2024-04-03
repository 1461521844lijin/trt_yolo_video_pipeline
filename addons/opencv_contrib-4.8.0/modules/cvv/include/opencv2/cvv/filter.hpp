#ifndef CVVISUAL_DEBUG_FILTER_HPP
#define CVVISUAL_DEBUG_FILTER_HPP

#include <string>

#include "opencv2/core.hpp"

#include "call_meta_data.hpp"
#include "debug_mode.hpp"

#ifdef CV_DOXYGEN
#define CVVISUAL_DEBUGMODE
#endif

namespace cvv
{

//! @addtogroup cvv
//! @{

namespace impl
{
// implementation outside API
CV_EXPORTS void debugFilter(cv::InputArray original, cv::InputArray result,
                 const CallMetaData &data, const char *description,
                 const char *view);
} // namespace impl

#ifdef CVVISUAL_DEBUGMODE
/**
 * @brief Use the debug-framework to compare two images (from which the second
 * is intended to be the result of
 * a filter applied to the first).
 */
static inline void
debugFilter(cv::InputArray original, cv::InputArray result,
            impl::CallMetaData metaData = impl::CallMetaData(),
            const char *description = nullptr, const char *view = nullptr)
{
	if (debugMode())
	{
		impl::debugFilter(original, result, metaData, description,
		                  view);
	}
}
/** @overload */
static inline void debugFilter(cv::InputArray original, cv::InputArray result,
                               impl::CallMetaData metaData,
                               const ::std::string &description,
                               const ::std::string &view = "")
{
	if (debugMode())
	{
		impl::debugFilter(original, result, metaData,
		                  description.c_str(), view.c_str());
	}
}
#else
static inline void debugFilter(cv::InputArray, cv::InputArray,
                               impl::CallMetaData = impl::CallMetaData(),
                               const char * = nullptr, const char * = nullptr)
{
}

static inline void debugFilter(cv::InputArray, cv::InputArray,
                               impl::CallMetaData, const ::std::string &,
                               const ::std::string &)
{
}
#endif

//! @}

} // namespace cvv

#endif
