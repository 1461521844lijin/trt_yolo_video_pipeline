// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_ARRAY_HPP__
#define __ZXING_COMMON_ARRAY_HPP__

#include "counted.hpp"

namespace zxing {

template <typename T>
class Array : public Counted {
protected:
public:
    std::vector<T> values_;
    Array() {}
    explicit Array(int n) : Counted(), values_(n, T()) {}
    Array(T const *ts, int n) : Counted(), values_(ts, ts + n) {}
    Array(T const *ts, T const *te) : Counted(), values_(ts, te) {}
    Array(T v, int n) : Counted(), values_(n, v) {}
    explicit Array(std::vector<T> &v) : Counted(), values_(v) {}
    Array(Array<T> &other) : Counted(), values_(other.values_) {}
    explicit Array(Array<T> *other) : Counted(), values_(other->values_) {}
    virtual ~Array() {}
    Array<T> &operator=(const Array<T> &other) {
        values_ = other.values_;
        return *this;
    }
    Array<T> &operator=(const std::vector<T> &array) {
        values_ = array;
        return *this;
    }
    T const &operator[](int i) const { return values_[i]; }
    T &operator[](int i) { return values_[i]; }
    int size() const { return values_.size(); }
    bool empty() const { return values_.size() == 0; }
    std::vector<T> const &values() const { return values_; }
    std::vector<T> &values() { return values_; }

    T *data() {
        // return values_.data();
        return &values_[0];
    }
    void append(T value) { values_.push_back(value); }
};

template <typename T>
class ArrayRef : public Counted {
private:
public:
    Array<T> *array_;
    ArrayRef() : array_(0) {}
    explicit ArrayRef(int n) : array_(0) { reset(new Array<T>(n)); }
    ArrayRef(T *ts, int n) : array_(0) { reset(new Array<T>(ts, n)); }
    explicit ArrayRef(Array<T> *a) : array_(0) { reset(a); }
    ArrayRef(const ArrayRef &other) : Counted(), array_(0) { reset(other.array_); }

    ~ArrayRef() {
        if (array_) {
            array_->release();
        }
        array_ = 0;
    }

    T const &operator[](int i) const { return (*array_)[i]; }

    T &operator[](int i) { return (*array_)[i]; }

    void reset(Array<T> *a) {
        if (a) {
            a->retain();
        }
        if (array_) {
            array_->release();
        }
        array_ = a;
    }
    void reset(const ArrayRef<T> &other) { reset(other.array_); }
    ArrayRef<T> &operator=(const ArrayRef<T> &other) {
        reset(other);
        return *this;
    }
    ArrayRef<T> &operator=(Array<T> *a) {
        reset(a);
        return *this;
    }

    Array<T> &operator*() const { return *array_; }

    Array<T> *operator->() const { return array_; }

    operator bool() const { return array_ != 0; }
    bool operator!() const { return array_ == 0; }

    T *data() { return array_->data(); }

    void clear() {
        T *ptr = array_->data();
        memset(ptr, 0, array_->size());
    }
    void append(T value) { array_->append(value); }
};

}  // namespace zxing

#endif  // __ZXING_COMMON_ARRAY_HPP__
