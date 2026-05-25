#pragma once
#include <vector>      // Needed for std::vector
#include <cstddef>     // For size_t
#include <cstdint>     // For uint8_t

namespace yolos::core {

class MemoryArena {
private:
    std::vector<uint8_t> buffer_;   // Raw byte buffer
    size_t offset_ = 0;

public:
    explicit MemoryArena(size_t size_in_bytes = 128ULL * 1024 * 1024) {
        buffer_.resize(size_in_bytes);
    }

    void* allocate(size_t size, size_t alignment = 64) {
        if (size == 0) return nullptr;

        size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + size > buffer_.size()) {
            return nullptr;
        }

        void* ptr = buffer_.data() + aligned_offset;
        offset_ = aligned_offset + size;

        return ptr;
    }

    void reset() {
        offset_ = 0;
    }

    size_t used() const { return offset_; }
    size_t capacity() const { return buffer_.size(); }

    // Helper for checking before allocation
    bool canAllocate(size_t size) const {
        if (size == 0) return true;
        size_t aligned = (offset_ + 63) & ~63;   // 64-byte alignment
        return aligned + size <= buffer_.size();
    }
};

} // namespace yolos::core