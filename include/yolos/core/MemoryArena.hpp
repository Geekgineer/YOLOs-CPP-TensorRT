#pragma once
#include <cstddef>
#include <vector>

namespace yolos::core {

class MemoryArena {
private:
    std::vector<char> buffer_;
    size_t offset_ = 0;

public:
    explicit MemoryArena(size_t size_in_bytes = 128ULL * 1024 * 1024) {  // 128 MB default
        buffer_.resize(size_in_bytes);
    }

    void* allocate(size_t size, size_t alignment = 64) {
        size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
        if (aligned + size > buffer_.size()) {
            return nullptr;
        }
        void* ptr = buffer_.data() + aligned;
        offset_ = aligned + size;
        return ptr;
    }

    void reset() {
        offset_ = 0;
    }

    size_t used() const { return offset_; }
};

} // namespace yolos::core