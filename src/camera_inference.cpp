/**
 * @file camera_inference.cpp
 * @brief Real-time object detection using YOLO TensorRT engines with camera input.
 *
 * Multi-threaded pipeline: capture → detect → display.
 * Press 'q' to quit.
 *
 * Usage:
 *   ./camera_inference <engine_path> [video_source] [labels_path]
 *
 * Author: YOLOs-TRT Team
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

#include <opencv2/highgui/highgui.hpp>

#include "yolos/tasks/detection.hpp"

using namespace yolos::det;

#include "tools/BoundedThreadSafeQueue.hpp"

int main(int argc, char* argv[]) {
    std::string labelsPath  = "../models/coco.names";
    std::string modelPath   = "../models/yolo11n.trt";
    std::string videoSource = "/dev/video0";

    if (argc > 1) modelPath   = argv[1];
    if (argc > 2) videoSource = argv[2];
    if (argc > 3) labelsPath  = argv[3];

    // Initialize the YOLO detector with TensorRT engine
    YOLODetector detector(modelPath, labelsPath);

    cv::VideoCapture cap;
    cap.open(videoSource, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera!\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    const size_t max_queue_size = 2;
    BoundedThreadSafeQueue<cv::Mat> frameQueue(max_queue_size);
    BoundedThreadSafeQueue<std::pair<cv::Mat, std::vector<Detection>>> processedQueue(max_queue_size);
    std::atomic<bool> stopFlag(false);

    // Producer thread: Capture frames
    std::thread producer([&]() {
        cv::Mat frame;
        while (!stopFlag.load() && cap.read(frame)) {
            if (!frameQueue.enqueue(frame))
                break;
        }
        frameQueue.set_finished();
    });

    // Consumer thread: Process frames
    std::thread consumer([&]() {
        cv::Mat frame;
        while (!stopFlag.load() && frameQueue.dequeue(frame)) {
            std::vector<Detection> detections = detector.detect(frame);
            if (!processedQueue.enqueue(std::make_pair(frame, detections)))
                break;
        }
        processedQueue.set_finished();
    });

    std::pair<cv::Mat, std::vector<Detection>> item;

#ifdef __APPLE__
    while (!stopFlag.load() && processedQueue.dequeue(item)) {
        cv::Mat displayFrame = item.first;
        detector.drawDetectionsWithMask(displayFrame, item.second);
        cv::imshow("Detections", displayFrame);
        if (cv::waitKey(1) == 'q') {
            stopFlag.store(true);
            frameQueue.set_finished();
            processedQueue.set_finished();
            break;
        }
    }
#else
    std::thread displayThread([&]() {
        while (!stopFlag.load() && processedQueue.dequeue(item)) {
            cv::Mat displayFrame = item.first;
            detector.drawDetectionsWithMask(displayFrame, item.second);
            cv::imshow("Detections", displayFrame);
            if (cv::waitKey(1) == 'q') {
                stopFlag.store(true);
                frameQueue.set_finished();
                processedQueue.set_finished();
                break;
            }
        }
    });
    displayThread.join();
#endif

    producer.join();
    consumer.join();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
