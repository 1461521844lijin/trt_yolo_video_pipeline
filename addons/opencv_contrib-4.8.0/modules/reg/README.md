OpenCV pixel-intensity based registration module
================================================

Author and maintainer: Alfonso Sanchez-Beato
                       alfonsosanchezbeato\_\_\_\_gmail.com

These classes implement a module for OpenCV for parametric image registration.
The implemented method is direct alignment, that is, it uses directly the pixel
values for calculating the registration between a pair of images, as opposed to
feature-based registration.  The implementation follows essentially the
corresponding part of the paper "Image Alignment and Stitching: A Tutorial",
from Richard Szeliski.

Feature based methods have some advantages over pixel based methods when we are
trying to register pictures that have been shoot under different lighting
conditions or exposition times, or when the images overlap only partially. On
the other hand, the main advantage of pixel-based methods when compared to
feature based methods is their better precision for some pictures (those shoot
under similar lighting conditions and that have a significative overlap), due to
the fact that we are using all the information available in the image, which
allows us to achieve subpixel accuracy. This is particularly important for
certain applications like multi-frame denoising or super-resolution.

In fact, pixel and feature registration methods can complement each other: an
application could first obtain a coarse registration using features and then
refine the registration using a pixel based method on the overlapping area of
the images. The code developed allows this use case.

The module implements classes derived from the abstract classes cv::reg::Map or
cv::reg::Mapper.  The former models a coordinate transformation between two
reference frames, while the later encapsulates a way of invoking a method that
calculates a Map between two images.  Although the objective has been to
implement pixel based methods, the module could be extended to support other
methods that can calculate transformations between images (feature methods,
optical flow, etc.).

Each class derived from Map implements a motion model, as follows:

* MapShift: Models a simple translation

* MapAffine: Models an affine transformation

* MapProject: Models a projective transformation
MapProject can also be used to model affine motion or translations, but some
operations on it are more costly, and that is the reason for defining the other
two classes.

The classes derived from Mapper are

* MapperGradShift: Gradient based alignment for calculating translations. It
produces a MapShift (two parameters that correspond to the shift vector).

* MapperGradEuclid: Gradient based alignment for euclidean motions, that is,
rotations and translations. It calculates three parameters (angle and shift
vector), although the result is stored in a MapAffine object for convenience.

* MapperGradSimilar: Gradient based alignment for calculating similarities,
which adds scaling to the euclidean motion. It calculates four parameters (two
for the anti-symmetric matrix and two for the shift vector), although the result
is stored in a MapAffine object for convenience.

* MapperGradAffine: Gradient based alignment for an affine motion model. The
number of parameters is six and the result is stored in a MapAffine object. 

* MapperGradProj: Gradient based alignment for calculating projective
transformations. The number of parameters is eight and the result is stored in a
MapProject object.

* MapperPyramid: It implements hyerarchical motion estimation using a Gaussian
pyramid. Its constructor accepts as argument any other object that implements
the Mapper interface, and it is that mapper the one called by MapperPyramid for
each scale of the pyramid.

If the motion between the images is not very small, the normal way of using
these classes is to create a MapperGrad\* object and use it as input to create a
MapperPyramid, which in turn is called to perform the calculation. However, if
the motion between the images is small enough, we can use directly the
MapperGrad\* classes. Another possibility is to use first a feature based method
to perform a coarse registration and then do a refinement through MapperPyramid
or directly a MapperGrad\* object. The "calculate" method of the mappers accepts
an initial estimation of the motion as input.

When deciding which MapperGrad to use we must take into account that mappers
with more parameters can handle more complex motions, but involve more
calculations and are therefore slower. Also, if we are confident on the motion
model that is followed by the sequence, increasing the number of parameters
beyond what we need will decrease the accuracy: it is better to use the least
number of degrees of freedom that we can.

In the file map_test.cpp some examples on how to use this module can be seen.
There is a test function for each MapperGrad\*. A motion is simulated on an input
image and then we register the moved image using a MapperPyramid created with
the right MapperGrad\*. The difference images of the pictures before and after
registering are displayed, and the ground truth parameters and the calculated
ones are printed. Additionally, two images from a real video are registered
using first SURF features and then MapperGradProj+MapperPyramid. The difference
between the images and the difference of the registered images using the two
methods are displayed. It can be seen in the differences shown that using a
pixel based difference we can achieve more accuracy.

