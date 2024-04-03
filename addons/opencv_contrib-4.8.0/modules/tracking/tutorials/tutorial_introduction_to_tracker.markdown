Introduction to OpenCV Tracker {#tutorial_introduction_to_tracker}
==============================

Goal
----

In this tutorial you will learn how to

-   Create a tracker object.
-   Use the roiSelector function to select a ROI from a given image.
-   Track a specific region in a given image.

Source Code
-----------

@include tracking/samples/tutorial_introduction_to_tracker.cpp

Explanation
-----------

-#  **Set up the input video**

    @snippet tracking/samples/tutorial_introduction_to_tracker.cpp help

    In this tutorial, you can choose between video or list of images for the program input.
    As written in the help, you should specify the input video as parameter of the program.
    If you want to use image list as input, the image list should have formatted numbering
    as shown in help. In the help, it means that the image files are numbered with 4 digits
    (e.g. the file naming will be 0001.jpg, 0002.jpg, and so on).

    You can find video samples in opencv_extra/testdata/cv/tracking
    <https://github.com/opencv/opencv_extra/tree/master/testdata/cv/tracking>

-#  **Declares the required variables**

    You need roi to record the bounding box of the tracked object. The value stored in this
    variable will be updated using the tracker object.

    @snippet tracking/samples/tutorial_introduction_to_tracker.cpp vars

    The frame variable is used to hold the image data from each frame of the input video or images list.

-#  **Creating a tracker object**

    @snippet tracking/samples/tutorial_introduction_to_tracker.cpp create

    There are at least 7 types of tracker algorithms that can be used:
    + MIL
    + BOOSTING
    + MEDIANFLOW
    + TLD
    + KCF
    + GOTURN
    + MOSSE

    Each tracker algorithm has their own advantages and disadvantages, please refer the documentation of @ref cv::Tracker for more detailed information.

-#  **Select the tracked object**

    @snippet tracking/samples/tutorial_introduction_to_tracker.cpp selectroi

    Using this function, you can select the bounding box of the tracked object using a GUI.
    With default parameters, the selection is started from the center of the box and a middle cross will be shown.

-#  **Initializing the tracker object**

    @snippet tracking/samples/tutorial_introduction_to_tracker.cpp init

    Any tracker algorithm should be initialized with the provided image data, and an initial bounding box of the tracked object.
    Make sure that the bounding box is valid (size more than zero) to avoid failure of the initialization process.

-#  **Update**

    @snippet tracking/samples/tutorial_introduction_to_tracker.cpp update

    This update function will perform the tracking process and pass the result to the roi variable.
