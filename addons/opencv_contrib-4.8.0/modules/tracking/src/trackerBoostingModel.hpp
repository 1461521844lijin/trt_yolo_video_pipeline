/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef __OPENCV_TRACKER_BOOSTING_MODEL_HPP__
#define __OPENCV_TRACKER_BOOSTING_MODEL_HPP__

namespace cv {
inline namespace tracking {
namespace impl {

/**
 * \brief Implementation of TrackerModel for BOOSTING algorithm
 */
class TrackerBoostingModel : public TrackerModel
{
 public:
  enum
  {
    MODE_POSITIVE = 1,    // mode for positive features
    MODE_NEGATIVE = 2,    // mode for negative features
    MODE_CLASSIFY = 3    // mode for classify step
  };
  /**
   * \brief Constructor
   * \param boundingBox The first boundingBox
   */
  TrackerBoostingModel( const Rect& boundingBox );

  /**
   * \brief Destructor
   */
  ~TrackerBoostingModel()
  {
  }
  ;

  /**
   * \brief Set the mode
   */
  void setMode( int trainingMode, const std::vector<Mat>& samples );

  /**
   * \brief Create the ConfidenceMap from a list of responses
   * \param responses The list of the responses
   * \param confidenceMap The output
   */
  void responseToConfidenceMap( const std::vector<Mat>& responses, ConfidenceMap& confidenceMap );

  /**
   * \brief return the selected weak classifiers for the detect
   * @return the selected weak classifiers
   */
  std::vector<int> getSelectedWeakClassifier();

 protected:
  void modelEstimationImpl( const std::vector<Mat>& responses ) CV_OVERRIDE;
  void modelUpdateImpl() CV_OVERRIDE;

 private:

  std::vector<Mat> currentSample;

  int mode;
};

}}}  // namespace

#endif
