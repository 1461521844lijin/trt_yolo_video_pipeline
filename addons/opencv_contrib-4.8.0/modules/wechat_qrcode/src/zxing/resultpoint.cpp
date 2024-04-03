// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../precomp.hpp"
#include "resultpoint.hpp"
#include "common/mathutils.hpp"

using zxing::common::MathUtils;

namespace zxing {

ResultPoint::ResultPoint() : posX_(0), posY_(0) {}

ResultPoint::ResultPoint(float x, float y) : posX_(x), posY_(y) {}

ResultPoint::ResultPoint(int x, int y) : posX_(float(x)), posY_(float(y)) {}

ResultPoint::~ResultPoint() {}

float ResultPoint::getX() const { return posX_; }

float ResultPoint::getY() const { return posY_; }

void ResultPoint::SetX(float fX) { posX_ = fX; }

void ResultPoint::SetY(float fY) { posY_ = fY; }

bool ResultPoint::equals(Ref<ResultPoint> other) {
    return (fabs(posX_ - other->getX()) <= 1e-6) && (fabs(posY_ - other->getY()) <= 1e-6);
}

/**
 * <p>Orders an array of three ResultPoints in an order [A,B,C] such that AB <
 * AC and BC < AC and the angle between BC and BA is less than 180 degrees.
 */
void ResultPoint::orderBestPatterns(std::vector<Ref<ResultPoint> > &patterns) {
    // Find distances between pattern centers
    float zeroOneDistance = distance(patterns[0]->getX(), patterns[1]->getX(), patterns[0]->getY(),
                                     patterns[1]->getY());
    float oneTwoDistance = distance(patterns[1]->getX(), patterns[2]->getX(), patterns[1]->getY(),
                                    patterns[2]->getY());
    float zeroTwoDistance = distance(patterns[0]->getX(), patterns[2]->getX(), patterns[0]->getY(),
                                     patterns[2]->getY());

    Ref<ResultPoint> pointA, pointB, pointC;
    // Assume one closest to other two is B; A and C will just be guesses at
    // first
    if (oneTwoDistance >= zeroOneDistance && oneTwoDistance >= zeroTwoDistance) {
        pointB = patterns[0];
        pointA = patterns[1];
        pointC = patterns[2];
    } else if (zeroTwoDistance >= oneTwoDistance && zeroTwoDistance >= zeroOneDistance) {
        pointB = patterns[1];
        pointA = patterns[0];
        pointC = patterns[2];
    } else {
        pointB = patterns[2];
        pointA = patterns[0];
        pointC = patterns[1];
    }

    // Use cross product to figure out whether A and C are correct or flipped.
    // This asks whether BC x BA has a positive z component, which is the
    // arrangement we want for A, B, C. If it's negative, then we've got it
    // flipped around and should swap A and C.
    if (crossProductZ(pointA, pointB, pointC) < 0.0f) {
        Ref<ResultPoint> temp = pointA;
        pointA = pointC;
        pointC = temp;
    }

    patterns[0] = pointA;
    patterns[1] = pointB;
    patterns[2] = pointC;
}

float ResultPoint::distance(Ref<ResultPoint> pattern1, Ref<ResultPoint> pattern2) {
    return MathUtils::distance(pattern1->posX_, pattern1->posY_, pattern2->posX_, pattern2->posY_);
}

float ResultPoint::distance(float x1, float x2, float y1, float y2) {
    float xDiff = x1 - x2;
    float yDiff = y1 - y2;
    return (float)sqrt((double)(xDiff * xDiff + yDiff * yDiff));
}

float ResultPoint::crossProductZ(Ref<ResultPoint> pointA, Ref<ResultPoint> pointB,
                                 Ref<ResultPoint> pointC) {
    float bX = pointB->getX();
    float bY = pointB->getY();
    return ((pointC->getX() - bX) * (pointA->getY() - bY)) -
           ((pointC->getY() - bY) * (pointA->getX() - bX));
}
}  // namespace zxing
