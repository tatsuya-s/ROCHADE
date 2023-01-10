#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "rochade.hpp"

int main(int argc, char** argv) {
    if (argc < 1) {
        std::cout << " Usage: chessboard [image file]" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(argv[1]);

    cv::Mat gray;
    cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    const cv::Size patternsize(9, 6);
    const int flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE;
    const bool patternfound = cv::findChessboardCorners(gray, patternsize, corners, flags);

    if (!patternfound) {
        std::cerr << "[Error] cv::findChessboardCorners()" << std::endl;
        return -1;
    }

    const cv::Size half_window_size(5, 5);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.001);

    // refile corners (OpenCV)
    std::vector<cv::Point2f> refined_opencv = corners;
    cv::Mat smoothed;
    cv::blur(gray, smoothed, half_window_size);
    cv::cornerSubPix(smoothed, refined_opencv, half_window_size, cv::Size(-1, -1), criteria);

    // refile corners (ROCHADE)
    std::vector<cv::Point2f> refined_rochade = corners;
    ROCHADE::saddleSubPix(gray, refined_rochade, half_window_size, criteria);

    cv::Mat result;
    img.copyTo(result);
    cv::circle(result, cv::Point(50, 42), 5, cv::Scalar(200, 0, 0), -1, cv::LINE_AA);
    cv::putText(result, "Initial", cv::Point(60, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(200, 0, 0), 2, cv::LINE_AA);
    cv::circle(result, cv::Point(50, 92), 5, cv::Scalar(0, 200, 0), -1, cv::LINE_AA);
    cv::putText(result, "OpenCV", cv::Point(60, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 0), 2, cv::LINE_AA);
    cv::circle(result, cv::Point(50, 142), 5, cv::Scalar(0, 0, 200), -1, cv::LINE_AA);
    cv::putText(result, "ROCHADE", cv::Point(60, 150), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 200), 2, cv::LINE_AA);
    for (size_t i = 0; i < corners.size(); ++i) {
        cv::circle(result, corners[i], 5, cv::Scalar(200, 0, 0), 1, cv::LINE_AA);
        cv::circle(result, refined_opencv[i], 5, cv::Scalar(0, 200, 0), 1, cv::LINE_AA);
        cv::circle(result, refined_rochade[i], 5, cv::Scalar(0, 0, 200), 1, cv::LINE_AA);
    }

    cv::imshow("result", result);
    cv::waitKey();

    return 0;
}