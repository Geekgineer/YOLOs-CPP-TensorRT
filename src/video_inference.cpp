/**
 * @file video_inference.cpp
 * @brief Object detection in a video stream using YOLO TensorRT engines.
 *
 * Multi-threaded pipeline: capture → detect → write.
 *
 * Usage:
 *   ./video_inference <engine_path> [video_path] [output_path] [labels_path]
 *
 * Author: YOLOs-TRT Team
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "yolos/tasks/detection.hpp"

using namespace yolos::det;

// Thread-safe queue implementation
template <typename T>
class SafeQueue {
public:
    SafeQueue() = default;

    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(std::move(t));
        c.notify_one();
    }

    bool dequeue(T& t) {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) {
            if (finished) return false;
            c.wait(lock);
        }
        t = std::move(q.front());
        q.pop();
        return true;
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(m);
        finished = true;
        c.notify_all();
    }

private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
    bool finished = false;
};

int main(int argc, char* argv[]) {
    std::string labelsPath  = "../models/coco.names";
    std::string videoPath   = "../data/dogs.mp4";
    std::string outputPath  = "../data/out_dogs.mp4";
    std::string modelPath   = "../models/yolo11n.trt";

    if (argc > 1) modelPath   = argv[1];
    if (argc > 2) videoPath   = argv[2];
    if (argc > 3) outputPath  = argv[3];
    if (argc > 4) labelsPath  = argv[4];

    // Initialize the YOLO detector with TensorRT engine
    YOLODetector detector(modelPath, labelsPath);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open or find the video file!\n";
        return -1;
    }

    int frameWidth  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps         = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc      = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

    cv::VideoWriter out(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
    if (!out.isOpened()) {
        std::cerr << "Error: Could not open the output video file for writing!\n";
        return -1;
    }

    SafeQueue<cv::Mat> frameQueue;
    SafeQueue<std::pair<int, cv::Mat>> processedQueue;
    std::atomic<bool> processingDone(false);

    // Capture thread
    std::thread captureThread([&]() {
        cv::Mat frame;
        while (cap.read(frame)) {
            frameQueue.enqueue(frame.clone());
        }
        frameQueue.setFinished();
    });

    // Processing thread
    std::thread processingThread([&]() {
        cv::Mat frame;
        int frameIndex = 0;
        while (frameQueue.dequeue(frame)) {
            std::vector<Detection> results = detector.detect(frame);
            detector.drawDetectionsWithMask(frame, results);
            processedQueue.enqueue(std::make_pair(frameIndex++, frame));
        }
        processedQueue.setFinished();
    });

    // Writing thread
    std::thread writingThread([&]() {
        std::pair<int, cv::Mat> processedFrame;
        while (processedQueue.dequeue(processedFrame)) {
            out.write(processedFrame.second);
        }
    });

    captureThread.join();
    processingThread.join();
    writingThread.join();

    cap.release();
    out.release();
    cv::destroyAllWindows();

    std::cout << "Video processing completed successfully." << std::endl;
    return 0;
}
