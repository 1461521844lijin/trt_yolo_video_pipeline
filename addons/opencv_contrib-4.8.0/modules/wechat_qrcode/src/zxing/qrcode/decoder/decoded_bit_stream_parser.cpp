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
#include "decoded_bit_stream_parser.hpp"
#include "../../common/stringutils.hpp"
#include "../../zxing.hpp"
#ifndef NO_ICONV_INSIDE
#include <iconv.h>
#endif
#include <iomanip>

#undef ICONV_CONST
#define ICONV_CONST const

#ifndef ICONV_CONST
#define ICONV_CONST /**/
#endif

using zxing::ErrorHandler;

// Add this to fix both Mac and Windows compilers
template <class T>
class sloppy {};

// convert between T** and const T**
template <class T>
class sloppy<T**> {
    T** t;

public:
    explicit sloppy(T** mt) : t(mt) {}
    explicit sloppy(const T** mt) : t(const_cast<T**>(mt)) {}

    operator T* *() const { return t; }
    operator const T* *() const { return const_cast<const T**>(t); }
};

using namespace std;
using namespace zxing;
using namespace zxing::qrcode;
using namespace zxing::common;

const char DecodedBitStreamParser::ALPHANUMERIC_CHARS[] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', ' ', '$', '%', '*', '+', '-', '.', '/', ':'};

// string DecodedBitStreamParser::outputCharset = "UTF-8";

namespace {
int GB2312_SUBSET = 1;
}

void DecodedBitStreamParser::append(std::string& result, string const& in,
                                    ErrorHandler& err_handler) {
    append(result, (char const*)in.c_str(), in.length(), err_handler);
}

void DecodedBitStreamParser::append(std::string& result, const char* bufIn, size_t nIn,
                                    ErrorHandler& err_handler) {
    // avoid null pointer exception
    if (err_handler.ErrCode() || bufIn == nullptr) return;
#ifndef NO_ICONV_INSIDE
    if (nIn == 0) {
        return;
    }
    iconv_t cd;
    // cout<<src<<endl;
    cd = iconv_open(StringUtils::UTF8, src);

    // iconv_t cd = iconv_open(StringUtils::GBK, src);
    if (cd == (iconv_t)-1) {
        result.append((const char*)bufIn, nIn);
        return;
    }

    const int maxOut = 4 * nIn + 1;
    char* bufOut = new char[maxOut];

    ICONV_CONST char* fromPtr = (ICONV_CONST char*)bufIn;
    size_t nFrom = nIn;
    char* toPtr = (char*)bufOut;
    size_t nTo = maxOut;

    while (nFrom > 0) {
        size_t oneway = iconv(cd, sloppy<char**>(&fromPtr), &nFrom, sloppy<char**>(&toPtr), &nTo);

        if (oneway == (size_t)(-1)) {
            iconv_close(cd);
            delete[] bufOut;
            err_handler = zxing::ReaderErrorHandler("error converting characters");
            return;
        }
    }
    iconv_close(cd);

    int nResult = maxOut - nTo;
    bufOut[nResult] = '\0';
    result.append((const char*)bufOut);
    delete[] bufOut;
#else
    result.append((const char*)bufIn, nIn);
#endif
}

void DecodedBitStreamParser::decodeHanziSegment(Ref<BitSource> bits_, string& result, int count,
                                                ErrorHandler& err_handler) {
    BitSource& bits(*bits_);
    // Don't crash trying to read more bits than we have available.
    if (count * 13 > bits.available()) {
        err_handler = zxing::FormatErrorHandler("decodeKanjiSegment");
        return;
    }

    // Each character will require 2 bytes. Read the characters as 2-byte pairs
    // and decode as GB2312 afterwards
    size_t nBytes = 2 * count;
    char* buffer = new char[nBytes];
    int offset = 0;
    while (count > 0) {
        // Each 13 bits encodes a 2-byte character
        int twoBytes = bits.readBits(13, err_handler);
        if (err_handler.ErrCode()) {
            delete[] buffer;
            return;
        }
        int assembledTwoBytes = ((twoBytes / 0x060) << 8) | (twoBytes % 0x060);
        if (assembledTwoBytes < 0x003BF) {
            // In the 0xA1A1 to 0xAAFE range
            assembledTwoBytes += 0x0A1A1;
        } else {
            // In the 0xB0A1 to 0xFAFE range
            assembledTwoBytes += 0x0A6A1;
        }
        buffer[offset] = (char)((assembledTwoBytes >> 8) & 0xFF);
        buffer[offset + 1] = (char)(assembledTwoBytes & 0xFF);
        offset += 2;
        count--;
    }
    // for(int i=0;i<nBytes;i++)
    // cout<<buffer[i]<<endl;
    append(result, buffer, nBytes, err_handler);
    if (err_handler.ErrCode()) {
        delete[] buffer;
        return;
    }

    delete[] buffer;
}

void DecodedBitStreamParser::decodeKanjiSegment(Ref<BitSource> bits, std::string& result, int count,
                                                ErrorHandler& err_handler) {
    // Each character will require 2 bytes. Read the characters as 2-byte pairs
    // and decode as Shift_JIS afterwards
    size_t nBytes = 2 * count;
    char* buffer = new char[nBytes];
    int offset = 0;
    while (count > 0) {
        // Each 13 bits encodes a 2-byte character

        int twoBytes = bits->readBits(13, err_handler);
        if (err_handler.ErrCode()) return;
        int assembledTwoBytes = ((twoBytes / 0x0C0) << 8) | (twoBytes % 0x0C0);
        if (assembledTwoBytes < 0x01F00) {
            // In the 0x8140 to 0x9FFC range
            assembledTwoBytes += 0x08140;
        } else {
            // In the 0xE040 to 0xEBBF range
            assembledTwoBytes += 0x0C140;
        }
        buffer[offset] = (char)(assembledTwoBytes >> 8);
        buffer[offset + 1] = (char)assembledTwoBytes;
        offset += 2;
        count--;
    }

    append(result, buffer, nBytes, err_handler);
    if (err_handler.ErrCode()) {
        delete[] buffer;
        return;
    }
    // cout<<buffer<<endl;
    delete[] buffer;
}

void DecodedBitStreamParser::decodeByteSegment(Ref<BitSource> bits_, string& result, int count,
                                               CharacterSetECI* currentCharacterSetECI,
                                               ArrayRef<ArrayRef<char> >& byteSegments,
                                               ErrorHandler& err_handler) {
    BitSource& bits(*bits_);
    // Don't crash trying to read more bits than we have available.
    int available = bits.available();
    // try to repair count data if count data is invalid
    if (count * 8 > available) {
        count = (available + 7) / 8;
    }
    size_t nBytes = count;

    ArrayRef<char> bytes_(nBytes);
    // issue https://github.com/opencv/opencv_contrib/issues/3478
    if (bytes_->empty())
        return;

    char* readBytes = &(*bytes_)[0];
    for (int i = 0; i < count; i++) {
        //    readBytes[i] = (char) bits.readBits(8);
        int readBits = available < 8 ? available : 8;
        readBytes[i] = (char)bits.readBits(readBits, err_handler);
    }
    if (err_handler.ErrCode()) return;
    // vector<string> encoding;
    string encoding;

    if (currentCharacterSetECI == 0) {
        // The spec isn't clear on this mode; see
        // section 6.4.5: t does not say which encoding to assuming
        // upon decoding. I have seen ISO-8859-1 used as well as
        // Shift_JIS -- without anything like an ECI designator to
        // give a hint.
        encoding = outputCharset;

    } else {
        // encoding .push_back(currentCharacterSetECI->name());
        encoding = currentCharacterSetECI->name();
    }
    // cout<<"encoding: "<<encoding<<endl;

    append(result, readBytes, nBytes, err_handler);
    if (err_handler.ErrCode()) return;

    byteSegments->values().push_back(bytes_);
}

void DecodedBitStreamParser::decodeNumericSegment(Ref<BitSource> bits, std::string& result,
                                                  int count, ErrorHandler& err_handler) {
    int nBytes = count;
    // char* bytes = new char[nBytes];
    ArrayRef<char> bytes = ArrayRef<char>(new Array<char>(nBytes));
    int i = 0;
    // Read three digits at a time
    while (count >= 3) {
        // Each 10 bits encodes three digits
        if (bits->available() < 10) {
            err_handler = zxing::ReaderErrorHandler("format exception");
            return;
        }
        int threeDigitsBits = bits->readBits(10, err_handler);
        if (err_handler.ErrCode()) return;
        if (threeDigitsBits >= 1000) {
            ostringstream s;
            s << "Illegal value for 3-digit unit: " << threeDigitsBits;
            err_handler = zxing::ReaderErrorHandler(s.str().c_str());
            return;
        }
        bytes[i++] = ALPHANUMERIC_CHARS[threeDigitsBits / 100];
        bytes[i++] = ALPHANUMERIC_CHARS[(threeDigitsBits / 10) % 10];
        bytes[i++] = ALPHANUMERIC_CHARS[threeDigitsBits % 10];
        count -= 3;
    }
    if (count == 2) {
        if (bits->available() < 7) {
            err_handler = zxing::ReaderErrorHandler("format exception");
            return;
        }
        // Two digits left over to read, encoded in 7 bits
        int twoDigitsBits = bits->readBits(7, err_handler);
        if (err_handler.ErrCode()) return;
        if (twoDigitsBits >= 100) {
            ostringstream s;
            s << "Illegal value for 2-digit unit: " << twoDigitsBits;
            err_handler = zxing::ReaderErrorHandler(s.str().c_str());
            return;
        }
        bytes[i++] = ALPHANUMERIC_CHARS[twoDigitsBits / 10];
        bytes[i++] = ALPHANUMERIC_CHARS[twoDigitsBits % 10];
    } else if (count == 1) {
        if (bits->available() < 4) {
            err_handler = zxing::ReaderErrorHandler("format exception");
            return;
        }
        // One digit left over to read
        int digitBits = bits->readBits(4, err_handler);
        if (err_handler.ErrCode()) return;
        if (digitBits >= 10) {
            ostringstream s;
            s << "Illegal value for digit unit: " << digitBits;
            err_handler = zxing::ReaderErrorHandler(s.str().c_str());
            return;
        }
        bytes[i++] = ALPHANUMERIC_CHARS[digitBits];
    }
    append(result, bytes->data(), nBytes, err_handler);
    if (err_handler.ErrCode()) return;
}

char DecodedBitStreamParser::toAlphaNumericChar(size_t value, ErrorHandler& err_handler) {
    if (value >= sizeof(DecodedBitStreamParser::ALPHANUMERIC_CHARS)) {
        err_handler = zxing::FormatErrorHandler("toAlphaNumericChar");
        return 0;
    }
    return ALPHANUMERIC_CHARS[value];
}

void DecodedBitStreamParser::decodeAlphanumericSegment(Ref<BitSource> bits_, string& result,
                                                       int count, bool fc1InEffect,
                                                       ErrorHandler& err_handler) {
    BitSource& bits(*bits_);
    ostringstream bytes;
    // Read two characters at a time
    while (count > 1) {
        if (bits.available() < 11) {
            err_handler = zxing::FormatErrorHandler("decodeAlphanumericSegment");
            return;
        }
        int nextTwoCharsBits = bits.readBits(11, err_handler);
        bytes << toAlphaNumericChar(nextTwoCharsBits / 45, err_handler);
        bytes << toAlphaNumericChar(nextTwoCharsBits % 45, err_handler);
        if (err_handler.ErrCode()) return;
        count -= 2;
    }
    if (count == 1) {
        // special case: one character left
        if (bits.available() < 6) {
            err_handler = zxing::FormatErrorHandler("decodeAlphanumericSegment");
            return;
        }
        bytes << toAlphaNumericChar(bits.readBits(6, err_handler), err_handler);
        if (err_handler.ErrCode()) return;
    }
    // See section 6.4.8.1, 6.4.8.2
    string s = bytes.str();
    if (fc1InEffect) {
        // We need to massage the result a bit if in an FNC1 mode:
        ostringstream r;
        for (size_t i = 0; i < s.length(); i++) {
            if (s[i] != '%') {
                r << s[i];
            } else {
                if (i < s.length() - 1 && s[i + 1] == '%') {
                    // %% is rendered as %
                    r << s[i++];
                } else {
                    // In alpha mode, % should be converted to FNC1 separator
                    // 0x1D
                    r << (char)0x1D;
                }
            }
        }
        s = r.str();
    }
    append(result, s, err_handler);
    if (err_handler.ErrCode()) return;
}

namespace {
int parseECIValue(BitSource& bits, ErrorHandler& err_handler) {
    int firstByte = bits.readBits(8, err_handler);
    if (err_handler.ErrCode()) return 0;
    if ((firstByte & 0x80) == 0) {
        // just one byte
        return firstByte & 0x7F;
    }
    if ((firstByte & 0xC0) == 0x80) {
        // two bytes
        int secondByte = bits.readBits(8, err_handler);
        if (err_handler.ErrCode()) return 0;
        return ((firstByte & 0x3F) << 8) | secondByte;
    }
    if ((firstByte & 0xE0) == 0xC0) {
        // three bytes
        int secondThirdBytes = bits.readBits(16, err_handler);
        if (err_handler.ErrCode()) return 0;
        return ((firstByte & 0x1F) << 16) | secondThirdBytes;
    }
    err_handler = zxing::FormatErrorHandler("parseECIValue");
    return 0;
}
}  // namespace

Ref<DecoderResult> DecodedBitStreamParser::decode(ArrayRef<char> bytes, Version* version,
                                                  ErrorCorrectionLevel const& ecLevel,
                                                  ErrorHandler& err_handler, int iVersion) {
    Ref<BitSource> bits_(new BitSource(bytes));
    BitSource& bits(*bits_);
    string result;
    result.reserve(50);
    Mode* mode = 0;
    string modeName;
    ArrayRef<ArrayRef<char> > byteSegments(0);

    CharacterSetECI* currentCharacterSetECI = 0;
    bool fc1InEffect = false;

    outputCharset = "UTF-8";
    do {
        // While still another segment to read...
        if (bits.available() < 4) {
            // OK, assume we're done. Really, a TERMINATOR mode should have been
            // recorded here
            mode = &Mode::TERMINATOR;
        } else {
            mode = &Mode::forBits(bits.readBits(4, err_handler),
                                  err_handler);  // mode is encoded by 4 bits
            if (err_handler.ErrCode()) return Ref<DecoderResult>();
        }

        if (mode != &Mode::TERMINATOR) {
            if ((mode == &Mode::FNC1_FIRST_POSITION) || (mode == &Mode::FNC1_SECOND_POSITION)) {
                // We do little with FNC1 except alter the parsed result a bit
                // according to the spec
                fc1InEffect = true;
            } else if (mode == &Mode::STRUCTURED_APPEND) {
                if (bits.available() < 16) {
                    err_handler = zxing::FormatErrorHandler("decode");
                    return Ref<DecoderResult>();
                }
                // not really supported; all we do is ignore it
                // Read next 8 bits (symbol sequence #) and 8 bits (parity
                // data), then continue
                bits.readBits(16, err_handler);
                if (err_handler.ErrCode()) return Ref<DecoderResult>();
            } else if (mode == &Mode::ECI) {
                // Count doesn't apply to ECI
                int value = parseECIValue(bits, err_handler);
                if (err_handler.ErrCode()) Ref<DecoderResult>();
                currentCharacterSetECI = CharacterSetECI::getCharacterSetECIByValueFind(value);
                if (currentCharacterSetECI == 0) {
                    err_handler = zxing::FormatErrorHandler("decode");
                    return Ref<DecoderResult>();
                }
            } else {
                // First handle Hanzi mode which does not start with character
                // count
                if (mode == &Mode::HANZI) {
                    // chinese mode contains a sub set indicator right after
                    // mode indicator
                    int subset = bits.readBits(4, err_handler);
                    int countHanzi =
                        bits.readBits(mode->getCharacterCountBits(version), err_handler);
                    if (err_handler.ErrCode()) return Ref<DecoderResult>();
                    if (subset == GB2312_SUBSET) {
                        decodeHanziSegment(bits_, result, countHanzi, err_handler);
                        if (err_handler.ErrCode()) Ref<DecoderResult>();
                        outputCharset = "GB2312";
                        modeName = mode->getName();
                    }
                } else {
                    // "Normal" QR code modes:
                    // How many characters will follow, encoded in this mode?
                    int count = bits.readBits(mode->getCharacterCountBits(version), err_handler);
                    if (err_handler.ErrCode()) return Ref<DecoderResult>();
                    if (mode == &Mode::NUMERIC) {
                        decodeNumericSegment(bits_, result, count, err_handler);
                        if (err_handler.ErrCode()) {
                            err_handler = zxing::FormatErrorHandler("decode");
                            return Ref<DecoderResult>();
                        }
                        modeName = mode->getName();
                    } else if (mode == &Mode::ALPHANUMERIC) {
                        decodeAlphanumericSegment(bits_, result, count, fc1InEffect, err_handler);
                        if (err_handler.ErrCode()) Ref<DecoderResult>();
                        modeName = mode->getName();
                    } else if (mode == &Mode::BYTE) {
                        decodeByteSegment(bits_, result, count, currentCharacterSetECI,
                                          byteSegments, err_handler);
                        if (err_handler.ErrCode()) {
                            err_handler = zxing::FormatErrorHandler("decode");
                            return Ref<DecoderResult>();
                        }
                        modeName = mode->getName();
                        // outputCharset = getResultCharset();
                    } else if (mode == &Mode::KANJI) {
                        // int countKanji =
                        // bits.readBits(mode->getCharacterCountBits(version));
                        // cout<<"countKanji: "<<countKanji<<endl;
                        // decodeKanjiSegment(bits_, result, countKanji);
                        decodeKanjiSegment(bits_, result, count, err_handler);
                        if (err_handler.ErrCode()) Ref<DecoderResult>();
                        modeName = mode->getName();
                    } else {
                        err_handler = zxing::FormatErrorHandler("decode");
                        return Ref<DecoderResult>();
                    }
                }
            }
        }
    } while (mode != &Mode::TERMINATOR);
    return Ref<DecoderResult>(new DecoderResult(bytes, Ref<String>(new String(result)),
                                                byteSegments, (string)ecLevel,
                                                (string)outputCharset, iVersion, modeName));
}
