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
#include "fast_window_binarizer.hpp"
using zxing::FastWindowBinarizer;


namespace {
const int BLOCK_SIZE = 6;
// const int BLOCK_SIZE = 8; // not as good as BLOCK_SIZE = 6
const float WINDOW_FRACTION = 0.13f;

static int min(int a, int b) { return a < b ? a : b; }

static int max(int a, int b) { return a > b ? a : b; }

}  // namespace

FastWindowBinarizer::FastWindowBinarizer(Ref<LuminanceSource> source)
    : GlobalHistogramBinarizer(source), matrix_(NULL), cached_row_(NULL) {
    width = source->getWidth();
    height = source->getHeight();
    int aw = width / BLOCK_SIZE;
    int ah = height / BLOCK_SIZE;

    int ah2 = ah;
    int ow2 = aw + 1;

    _luminancesInt = new int[width * height];
    _blockTotals = new int[ah * aw];
    _totals = new int[(ah + 1) * (aw + 1)];
    _rowTotals = new int[ah2 * ow2];

    _internal = new unsigned int[(height + 1) * (width + 1)];
}

FastWindowBinarizer::~FastWindowBinarizer() {
    delete[] _totals;
    delete[] _blockTotals;
    delete[] _luminancesInt;
    delete[] _rowTotals;

    delete[] _internal;
}

Ref<Binarizer> FastWindowBinarizer::createBinarizer(Ref<LuminanceSource> source) {
    return Ref<Binarizer>(new FastWindowBinarizer(source));
}

/**
 * Calculates the final BitMatrix once for all requests. This could be called
 * once from the constructor instead, but there are some advantages to doing it
 * lazily, such as making profiling easier, and not doing heavy lifting when
 * callers don't expect it.
 */
Ref<BitMatrix> FastWindowBinarizer::getBlackMatrix(ErrorHandler& err_handler) {
    if (!matrix0_) {
        binarizeImage1(err_handler);
        if (err_handler.ErrCode()) return Ref<BitMatrix>();
    }

    return Binarizer::getBlackMatrix(err_handler);
}

/**
 * Calculate black row from BitMatrix
 * If BitMatrix has been calculated then just get the row
 * If BitMatrix has not been calculated then call getBlackMatrix first
 */

Ref<BitArray> FastWindowBinarizer::getBlackRow(int y, Ref<BitArray> row,
                                               ErrorHandler& err_handler) {
    if (!matrix0_) {
        binarizeImage1(err_handler);
        if (err_handler.ErrCode()) return Ref<BitArray>();
    }
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackRow(y, row, err_handler);
}

void FastWindowBinarizer::calcBlockTotals(int* luminancesInt, int* output, int aw, int ah) {
    for (int by = 0; by < ah; by++) {
        int ey = (by + 1) * BLOCK_SIZE;
        for (int bx = 0; bx < aw; bx++) {
            int t = 0;
            for (int y = by * BLOCK_SIZE; y < ey; y++) {
                int offset = y * width + bx * BLOCK_SIZE;
                int ex = offset + BLOCK_SIZE;
                for (; offset < ex; offset++) {
                    // int v = luminancesInt[offset] & 0xff;
                    t += luminancesInt[offset];
                }
            }
            output[by * aw + bx] = t;
        }
    }
}

void FastWindowBinarizer::cumulative(int* data, int* output, int _width, int _height) {
    int ah = _height;
    int aw = _width;
    int ow = _width + 1;
    // int[][] totals = new int[ah + 1][aw + 1];
    // int* rowTotals = new int[ah*ow];

    for (int y = 0; y < ah; y++) {
        int* row = _rowTotals + (y * ow);
        int* rowdata = data + (y * aw);
        int t = 0;
        row[0] = t;

        for (int x = 0; x < aw; x++) {
            t += rowdata[x];
            row[x + 1] = t;
        }
    }

    for (int x = 0; x <= aw; x++) {
        output[x] = 0;  // First row
        int t = 0;

        for (int y = 0; y < ah; y++) {
            t += _rowTotals[y * ow + x];
            output[(y + 1) * ow + x] = t;
        }
    }
}

void FastWindowBinarizer::fastIntegral(const unsigned char* inputMatrix,
                                       unsigned int* outputMatrix) {
    // memset(outputMatrix,0,sizeof(int)*(height+1)*(width+1));
    // unsigned int *columnSum = new unsigned int[width]; // sum of each column
    // calculate integral of the first line
    outputMatrix[0] = outputMatrix[width + 1] = 0;
    for (int i = 0; i < width; i++) {
        // columnSum[i]=inputMatrix[i];
        outputMatrix[i + 1] = 0;
        outputMatrix[width + 1 + i + 1] = outputMatrix[width + 1 + i] + inputMatrix[i];
    }
    for (int i = 1; i < height; i++) {
        const unsigned char* psi = inputMatrix + i * width;
        unsigned int* pdi = outputMatrix + (i + 1) * (width + 1);
        // first column of each line
        pdi[0] = 0;
        pdi[1] = psi[0];
        int row_sum = psi[0];
        // other columns
        for (int j = 1; j < width; j++) {
            row_sum += psi[j];
            pdi[j + 1] = pdi[j + 1 - width - 1] + row_sum;
        }
    }
    return;
}

int FastWindowBinarizer::binarizeImage1(ErrorHandler& err_handler) {
    LuminanceSource& source = *getLuminanceSource();
    Ref<BitMatrix> matrix(new BitMatrix(width, height, err_handler));
    if (err_handler.ErrCode()) return -1;

    ArrayRef<char> localLuminances = source.getMatrix();

    unsigned char* src = (unsigned char*)localLuminances->data();
    unsigned char* dst = matrix->getPtr();
    fastWindow(src, dst, err_handler);
    if (err_handler.ErrCode()) return -1;

    matrix0_ = matrix;
    return 0;
}

void FastWindowBinarizer::fastWindow(const unsigned char* src, unsigned char* dst,
                                     ErrorHandler& err_handler) {
    int r = (int)(min(width, height) * WINDOW_FRACTION / BLOCK_SIZE / 2 + 1);
    const int NEWH_BLOCK_SIZE = BLOCK_SIZE * r;
    if (height < NEWH_BLOCK_SIZE || width < NEWH_BLOCK_SIZE) {
        matrix_ = GlobalHistogramBinarizer::getBlackMatrix(err_handler);
        return;
    }
    const unsigned char* _img = src;
    fastIntegral(_img, _internal);
    int aw = width / BLOCK_SIZE;
    int ah = height / BLOCK_SIZE;
    memset(dst, 0, sizeof(char) * height * width);
    for (int ai = 0; ai < ah; ai++) {
        int top = max(0, ((ai - r + 1) * BLOCK_SIZE));
        int bottom = min(height, (ai + r) * BLOCK_SIZE);
        unsigned int* pt = _internal + top * (width + 1);
        unsigned int* pb = _internal + bottom * (width + 1);
        for (int aj = 0; aj < aw; aj++) {
            int left = max(0, (aj - r + 1) * BLOCK_SIZE);
            int right = min(width, (aj + r) * BLOCK_SIZE);
            unsigned int block = pb[right] + pt[left] - pt[right] - pb[left];
            int pixels = (bottom - top) * (right - left);
            int avg = (int)block / pixels;
            for (int bi = ai * BLOCK_SIZE; bi < height && bi < (ai + 1) * BLOCK_SIZE; bi++) {
                const unsigned char* psi = src + bi * width;
                unsigned char* pdi = dst + bi * width;
                for (int bj = aj * BLOCK_SIZE; bj < width && bj < (aj + 1) * BLOCK_SIZE; bj++) {
                    if ((int)psi[bj] < avg)
                        pdi[bj] = 1;
                    else
                        pdi[bj] = 0;
                }
            }
        }
    }
    // delete [] _internal;
    return;
}

int FastWindowBinarizer::binarizeImage0(ErrorHandler& err_handler) {
    // if (matrix_) {
    //	return matrix_;
    //}

    LuminanceSource& source = *getLuminanceSource();
    if (width >= BLOCK_SIZE && height >= BLOCK_SIZE) {
        int r = (int)(min(width, height) * WINDOW_FRACTION / BLOCK_SIZE / 2 + 1);

        int aw = width / BLOCK_SIZE;
        int ah = height / BLOCK_SIZE;
        int ow = aw + 1;

        ArrayRef<char> _luminances = source.getMatrix();

        // Get luminances for int value first
        for (int i = 0; i < width * height; i++) {
            _luminancesInt[i] = _luminances[i] & 0xff;
        }

        calcBlockTotals(_luminancesInt, _blockTotals, aw, ah);

        cumulative(_blockTotals, _totals, aw, ah);

        Ref<BitMatrix> newMatrix(new BitMatrix(width, height, err_handler));
        if (err_handler.ErrCode()) return -1;
        unsigned char* newimg = newMatrix->getPtr();
        for (int by = 0; by < ah; by++) {
            int top = max(0, by - r + 1);
            int bottom = min(ah, by + r);

            for (int bx = 0; bx < aw; bx++) {
                int left = max(0, bx - r + 1);
                int right = min(aw, bx + r);

                int block = _totals[bottom * ow + right] + _totals[top * ow + left] -
                            _totals[top * ow + right] - _totals[bottom * ow + left];

                int pixels = (bottom - top) * (right - left) * BLOCK_SIZE * BLOCK_SIZE;
                int avg = block / pixels;

                for (int y = by * BLOCK_SIZE; y < (by + 1) * BLOCK_SIZE; y++) {
                    // int offset = y*width;
                    int* plumint = _luminancesInt + y * width;
                    unsigned char* pn = newimg + y * width;
                    for (int x = bx * BLOCK_SIZE; x < (bx + 1) * BLOCK_SIZE; x++) {
                        // int pixel = luminances[y*width + x] & 0xff;
                        // if(plumint[x] < avg)
                        // newMatrix->set(x, y);
                        if (plumint[x] < avg)
                            pn[x] = 1;
                        else
                            pn[x] = 0;
                    }
                }
            }
        }
        // delete[] data;
        matrix_ = newMatrix;
    } else {
        // If the image is too small, fall back to the global histogram
        // approach.
        matrix_ = GlobalHistogramBinarizer::getBlackMatrix(err_handler);
    }

    return 0;
}
