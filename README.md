# ROCHADE

A header-only, C++ implementation of the corner refinement based on [ROCHADE](https://link.springer.com/chapter/10.1007/978-3-319-10593-2_50)

## Installation

Simply add OpenCV and `rochade.hpp` to your project

## Usage

```cpp
#include "rochade.hpp"
```
```cpp
cv::Mat gray;
std::vector<cv::Point2f> corners;

// detect corners from graycale image
// ...

// call ROCHADE::saddleSubPix() intead of cv::cornerSubPix()
const cv::Size half_window_size(5, 5);
const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.001);
ROCHADE::saddleSubPix(gray, corners, half_window_size, criteria);
```

## License

This code is licensed under the LGPL v2.1 license. For more information, please see COPYING file.

## Acknowledgments

This software uses code from `PolynomialFit.h` in [Deltille detector](https://github.com/deltille/detector)

```
Copyright (C) 2017-present, Facebook, Inc.
SPDX-License-Identifier: LGPL-2.1

Created on: Nov 29, 2016
    Author: mperdoch
```