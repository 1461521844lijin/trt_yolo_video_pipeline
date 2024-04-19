#ifndef CVVISUAL_TRANLSATION_MATCH_VIEW
#define CVVISUAL_TRANLSATION_MATCH_VIEW

#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

#include "../qtutil/matchview/matchmanagement.hpp"
#include "../qtutil/matchview/keypointmanagement.hpp"
#include "match_view.hpp"
namespace cvv
{
namespace view
{
/**
 * @brief this view shows the matches as tranlations line/arrows.
 *
 */

class TranslationMatchView : public MatchView
{

	Q_OBJECT
      public:
	/**
	 * @brief the constructor
	 * @param lefKeyPoints (queryindx) the keypoint from the left image
	 * @param rightKeyPoint (trainIdx/imIdx) the keypoints from the right
	 *Image
	 * @param matches the matches between the images
	 * @param usetrainIdx if true the trainIdx will be taken for
	 *rightKeyPoint if false
	 *	the imIdx will be taken
	 * @param parent the parent widget
	 */
	TranslationMatchView(std::vector<cv::KeyPoint> leftKeyPoints,
			     std::vector<cv::KeyPoint> rightKeyPoints,
			     std::vector<cv::DMatch> matches, cv::Mat leftIm,
			     cv::Mat rightIm, bool usetrainIdx = true,
			     QWidget *parent = nullptr);

	/**
	 * @brief Short constructor.
	 * @param call from which the data for the view is taken.
	 * @param parent of this QWidget.
	 */
	TranslationMatchView(const impl::MatchCall &call,
			     QWidget *parent = nullptr)
	    : TranslationMatchView{
		      call.keyPoints1(), call.keyPoints2(),
		      call.matches(),    call.img1(),
		      call.img2(),       call.usesTrainDescriptor(),
		      parent
}
	{}

	virtual std::vector<cv::DMatch> getMatchSelection() override
	{
		return matchManagment_->getCurrentSelection();
	}

    virtual std::vector<cv::KeyPoint> getKeyPointSelection() CV_OVERRIDE
	{
		return keyManagment_->getCurrentSelection();
	}

public slots:

    virtual void setMatchSelection(std::vector<cv::DMatch> selection) CV_OVERRIDE
	{
		matchManagment_->setSelection(selection);
	}

    virtual void setKeyPointSelection(std::vector<cv::KeyPoint> selection) CV_OVERRIDE
	{
		keyManagment_->setSelection(selection);
	}

private slots:

	void updateMousHoverOver(QPointF pt,QString str,bool){
		emit updateRightFoooter(QString("%1/%2 RGB:%3").arg(pt.x()).arg(pt.y()).arg(str));
	}

private:
	qtutil::MatchManagement *matchManagment_;
	qtutil::KeyPointManagement *keyManagment_;
};
}
}
#endif
