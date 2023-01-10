/**
* Copyright (C) 2017-present, Facebook, Inc.
* Copyright (C) 2023-present, Tatsuya Sakuma
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef ROCHADE_HPP_
#define ROCHADE_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace ROCHADE {
    int createConeSmoothingKernel(
        const cv::Size& half_window_size,
        cv::Mat& smoothing_kernel,
        cv::Mat& mask
    ) {
        const cv::Size window_size(half_window_size.width * 2 + 1, half_window_size.height * 2 + 1);
        smoothing_kernel.create(window_size, CV_64FC1);
        mask.create(window_size, CV_8UC1);

        const double max_val = (std::min)(half_window_size.width, half_window_size.height) + 1;
        double sum = 0.0;
        double* w = smoothing_kernel.ptr<double>(0);
        uint8_t* m = mask.ptr<uint8_t>(0);

        // cone kernel
        int non_zero_count = 0;
        for (int y = -half_window_size.height; y <= half_window_size.height; ++y) {
            for (int x = -half_window_size.width; x <= half_window_size.width; ++x) {
                *w = max_val - std::sqrt(x * x + y * y);
                if (*w > 0) {
                    *m = 255;
                    non_zero_count++;
                }
                else {
                    *w = 0.0;
                    *m = 0;
                }
                sum += *w;
                w++;
                m++;
            }
        }

        // scale kernel
        smoothing_kernel /= sum;
        return non_zero_count;
    }

    bool interpolatePatch(
        const double x,
        const double y,
        const cv::Size& half_window_size,
        const cv::Mat& src,
        const cv::Mat& mask,
        cv::Mat& b_vec
    ) {
        if (x > half_window_size.width + 1 && x < src.cols - (half_window_size.width + 2) &&
            y > half_window_size.height + 1 && y < src.rows - (half_window_size.height + 2)) {
            const int x0 = static_cast<int>(x);
            const int y0 = static_cast<int>(y);
            const double xw = x - x0;
            const double yw = y - y0;

            // precompute bilinear interpolation weights
            const double w00 = (1.0 - xw) * (1.0 - yw);
            const double w01 = xw * (1.0 - yw);
            const double w10 = (1.0 - xw) * yw;
            const double w11 = xw * yw;

            // fit to local neighborhood = b vector...
            const uint8_t* v = mask.ptr<const uint8_t>(0);
            double* m = b_vec.ptr<double>(0);
            double mn = (std::numeric_limits<double>::max)();
            double mx = std::numeric_limits<double>::lowest();
            for (int wy = -half_window_size.height; wy <= half_window_size.height; ++wy) {
                const double* im00 = src.ptr<double>(y0 + wy);
                const double* im10 = src.ptr<double>(y0 + wy + 1);
                for (int wx = -half_window_size.width; wx <= half_window_size.width; ++wx) {
                    if (*v > 0) {
                        const int col0 = x0 + wx;
                        const int col1 = col0 + 1;
                        const double val = im00[col0] * w00 + im00[col1] * w01 + im10[col0] * w10 + im10[col1] * w11;
                        *(m++) = val;
                        mn = (std::min)(val, mn);
                        mx = (std::max)(val, mx);
                    }
                    v++;
                }
            }
            if (mx - mn > 1.0 / 255) {
                return true;
            }
        }
        return false;
    }

    void saddleSubPix(
        const cv::Mat& gray,
        std::vector<cv::Point2d>& corners,
        const cv::Size& half_window_size,
        const cv::TermCriteria& criteria
    ) {
        // create cone smooth kernel
        cv::Mat smoothing_kernel;
        cv::Mat mask;
        const int non_zero_count = createConeSmoothingKernel(half_window_size, smoothing_kernel, mask);

        // smoothing
        cv::Mat smoothed;
        cv::filter2D(gray, smoothed, CV_64FC1, smoothing_kernel);

        // compute SVD
        cv::Mat w, u, vt;
        {
            cv::Mat A(non_zero_count, 6, CV_64FC1);
            double* a = A.ptr<double>(0);
            uint8_t* m = mask.ptr<uint8_t>(0);
            for (int y = -half_window_size.height; y <= half_window_size.height; ++y) {
                for (int x = -half_window_size.width; x <= half_window_size.width; ++x) {
                    if (*m > 0) {
                        a[0] = x * x;
                        a[1] = y * y;
                        a[2] = x * y;
                        a[3] = y;
                        a[4] = x;
                        a[5] = 1;
                        a += 6;
                    }
                    m++;
                }
            }

            // compute w, u, and vt
            cv::SVDecomp(A, w, u, vt, cv::SVD::FULL_UV);
        }

        int max_iters = 100;
        double eps = 0.0;
        switch (criteria.type) {
            case cv::TermCriteria::COUNT:
                max_iters = criteria.maxCount;
                break;
            case cv::TermCriteria::EPS:
                eps = criteria.epsilon;
                break;
            case cv::TermCriteria::COUNT | cv::TermCriteria::EPS:
                max_iters = criteria.maxCount;
                eps = criteria.epsilon;
                break;
            default:
                break;
        }

        cv::Mat b(non_zero_count, 1, CV_64FC1);

        const std::vector<cv::Point2d> initial = corners;
        std::vector<cv::Point2d>& refined = corners;

        for (size_t idx = 0; idx < refined.size(); ++idx) {
            cv::Point2d& pt = refined[idx];
            bool is_converged = true;
            for (int it = 0; it < max_iters; ++it) {
                if (interpolatePatch(pt.x, pt.y, half_window_size, smoothed, mask, b)) {
                    // fit quadric to surface by using SVD results
                    cv::Mat p;
                    cv::SVD::backSubst(w, u, vt, b, p);

                    // k5, k4, k3, k2, k1, k0
                    // 0 , 1 , 2 , 3 , 4 , 5
                    double* r = p.ptr<double>(0);
                    const double det = 4.0 * r[0] * r[1] - r[2] * r[2]; // 4.0 * k5 * k4 - k3 * k3

                    // check if it is still a saddle point
                    if (det > 0) {
                        is_converged = false;
                        break;
                    }

                    // compute the new location
                    const double dx = (-2.0 * r[1] * r[4] + r[2] * r[3]) / det; // - 2 * k4 * k1 +     k3 * k2
                    const double dy = (r[2] * r[4] - 2.0 * r[0] * r[3]) / det;  //       k3 * k1 - 2 * k5 * k2
                    pt.x += dx;
                    pt.y += dy;

                    if (eps > std::abs(dx) &&
                        eps > std::abs(dy)) {
                        // converged
                        is_converged = true;
                        break;
                    }
                    // check for divergence due to departure out of convergence region or
                    // point type change
                    if (std::abs(pt.x - initial[idx].x) > half_window_size.width ||
                        std::abs(pt.y - initial[idx].y) > half_window_size.height) {
                        is_converged = false;
                        break;
                    }
                }
                else {
                    is_converged = false;
                    break;
                }
            }
            if (!is_converged) {
                pt.x = std::numeric_limits<double>::infinity();
                pt.y = std::numeric_limits<double>::infinity();
            }
        }
    }

    void saddleSubPix(
        const cv::Mat& gray,
        std::vector<cv::Point2f>& corners_2f,
        const cv::Size& half_window_size,
        const cv::TermCriteria& criteria
    ) {
        std::vector<cv::Point2d> corners_2d(corners_2f.size());
        std::transform(corners_2f.begin(), corners_2f.end(), corners_2d.begin(), [](const cv::Point2f& p) { return cv::Point2d(p.x, p.y); });
        saddleSubPix(gray, corners_2d, half_window_size, criteria);
        std::transform(corners_2d.begin(), corners_2d.end(), corners_2f.begin(), [](const cv::Point2d& p) { return cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)); });
    }
}

#endif // ROCHADE_HPP_
