// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../../../precomp.hpp"
#include "simple_adaptive_binarizer.hpp"

using zxing::SimpleAdaptiveBinarizer;


SimpleAdaptiveBinarizer::SimpleAdaptiveBinarizer(Ref<LuminanceSource> source)
    : GlobalHistogramBinarizer(source) {
    filtered = false;
}

SimpleAdaptiveBinarizer::~SimpleAdaptiveBinarizer() {}

// Applies simple sharpening to the row data to improve performance of the 1D
// readers.
Ref<BitArray> SimpleAdaptiveBinarizer::getBlackRow(int y, Ref<BitArray> row,
                                                   ErrorHandler &err_handler) {
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_) {
        binarizeImage0(err_handler);
        if (err_handler.ErrCode()) return Ref<BitArray>();
    }
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackRow(y, row, err_handler);
}

// Does not sharpen the data, as this call is intended to only be used by 2D
// readers.
Ref<BitMatrix> SimpleAdaptiveBinarizer::getBlackMatrix(ErrorHandler &err_handler) {
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_) {
        binarizeImage0(err_handler);
        if (err_handler.ErrCode()) return Ref<BitMatrix>();
    }

    // First call binarize image in child class to get matrix0_ and binCache
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackMatrix(err_handler);
}

using namespace std;

int SimpleAdaptiveBinarizer::binarizeImage0(ErrorHandler &err_handler) {
    LuminanceSource &source = *getLuminanceSource();

    Ref<BitMatrix> matrix(new BitMatrix(width, height, err_handler));
    if (err_handler.ErrCode()) return -1;

    ArrayRef<char> localLuminances = source.getMatrix();

    unsigned char *src = (unsigned char *)localLuminances->data();
    unsigned char *dst = matrix->getPtr();

    qrBinarize(src, dst);

    matrix0_ = matrix;

    return 0;
}

/*A simplified adaptive thresholder.
  This compares the current pixel value to the mean value of a (large) window
   surrounding it.*/
int SimpleAdaptiveBinarizer::qrBinarize(const unsigned char *src, unsigned char *dst) {
    unsigned char *mask = dst;

    if (width > 0 && height > 0) {
        unsigned *col_sums;
        int logwindw;
        int logwindh;
        int windw;
        int windh;
        int y0offs;
        int y1offs;
        unsigned g;
        int x;
        int y;
        /*We keep the window size fairly large to ensure it doesn't fit
           completely inside the center of a finder pattern of a version 1 QR
           code at full resolution.*/
        for (logwindw = 4; logwindw < 8 && (1 << logwindw) < ((width + 7) >> 3); logwindw++)
            ;
        for (logwindh = 4; logwindh < 8 && (1 << logwindh) < ((height + 7) >> 3); logwindh++)
            ;
        windw = 1 << logwindw;
        windh = 1 << logwindh;

        int logwinds = (logwindw + logwindh);

        col_sums = (unsigned *)malloc(width * sizeof(*col_sums));
        /*Initialize sums down each column.*/
        for (x = 0; x < width; x++) {
            g = src[x];
            col_sums[x] = (g << (logwindh - 1)) + g;
        }
        for (y = 1; y < (windh >> 1); y++) {
            y1offs = min(y, height - 1) * width;
            for (x = 0; x < width; x++) {
                g = src[y1offs + x];
                col_sums[x] += g;
            }
        }
        for (y = 0; y < height; y++) {
            unsigned m;
            int x0;
            int x1;
            /*Initialize the sum over the window.*/
            m = (col_sums[0] << (logwindw - 1)) + col_sums[0];
            for (x = 1; x < (windw >> 1); x++) {
                x1 = min(x, width - 1);
                m += col_sums[x1];
            }

            int offset = y * width;

            for (x = 0; x < width; x++) {
                /*Perform the test against the threshold T = (m/n)-D,
                   where n=windw*windh and D=3.*/
                g = src[offset + x];
                mask[offset + x] = ((g + 3) << (logwinds) < m);
                /*Update the window sum.*/
                if (x + 1 < width) {
                    x0 = max(0, x - (windw >> 1));
                    x1 = min(x + (windw >> 1), width - 1);
                    m += col_sums[x1] - col_sums[x0];
                }
            }
            /*Update the column sums.*/
            if (y + 1 < height) {
                y0offs = max(0, y - (windh >> 1)) * width;
                y1offs = min(y + (windh >> 1), height - 1) * width;
                for (x = 0; x < width; x++) {
                    col_sums[x] -= src[y0offs + x];
                    col_sums[x] += src[y1offs + x];
                }
            }
        }
        free(col_sums);
    }

    return 1;
}

Ref<Binarizer> SimpleAdaptiveBinarizer::createBinarizer(Ref<LuminanceSource> source) {
    return Ref<Binarizer>(new SimpleAdaptiveBinarizer(source));
}
