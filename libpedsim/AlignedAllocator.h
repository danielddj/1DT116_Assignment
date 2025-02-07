#pragma once

#include <vector>
#include <cstdlib>       // posix_memalign, free
#include <stdexcept>     // std::bad_alloc

inline void* alignedMalloc(std::size_t alignment, std::size_t size)
{
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        ptr = nullptr;
    return ptr;
}

inline void alignedFree(void* ptr)
{
    free(ptr);
}

template <typename T, std::size_t Alignment>
struct AlignedAllocator
{
    using value_type = T;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n)
    {
        std::size_t size = n * sizeof(T);
        void* ptr = alignedMalloc(Alignment, size);
        if (!ptr) throw std::bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t /*n*/) noexcept
    {
        alignedFree(p);
    }
};

template <typename T, std::size_t A, typename U, std::size_t B>
inline bool operator==(const AlignedAllocator<T,A>&, const AlignedAllocator<U,B>&)
{
    return A == B;
}
template <typename T, std::size_t A, typename U, std::size_t B>
inline bool operator!=(const AlignedAllocator<T,A>&, const AlignedAllocator<U,B>&)
{
    return A != B;
}

using FloatVectorAligned16 = std::vector<float, AlignedAllocator<float, 16>>;
using IntVectorAligned16   = std::vector<int,   AlignedAllocator<int, 16>>;

using FloatVectorAligned32 = std::vector<float, AlignedAllocator<float, 32>>;
using IntVectorAligned32   = std::vector<int,   AlignedAllocator<int, 32>>;
