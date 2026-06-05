#include <iostream>
#include "yolos/tasks/detection.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <engine> <labels> <image>\n";
        return -1;
    }

    std::cout << "=== YOLOs-TRT (Dummy Mode) ===\n";
    std::cout << "Engine: " << argv[1] << "\n";
    std::cout << "Labels: " << argv[2] << "\n";
    std::cout << "Image : " << argv[3] << "\n\n";

    std::cout << "✅ Program is working!\n";
    std::cout << "Dummy detection: Found 3 objects on the dog image.\n";
    std::cout << "   - Person (conf 0.92)\n";
    std::cout << "   - Dog (conf 0.89)\n";
    std::cout << "   - Ball (conf 0.76)\n";

    std::cout << "\nTo get real detection, you need a valid .trt file.\n";

    return 0;
}
