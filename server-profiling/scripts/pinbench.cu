// copy one slot-sized KV state GPU -> host: today (vector resize + copy), no zero-fill, reused buffer, reused pinned buffer,
// plus: reused buffer that is zero-filled again before each copy, and fresh buffers backed by 2 MiB pages (MADV_HUGEPAGE)
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <sys/mman.h>

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? atoll(argv[1]) : 1560;
    const int nchunks = argc > 2 ? atoi(argv[2]) : 96; // llama.cpp copies K and V per layer: 48 x 2
    const size_t n = mib << 20, chunk = n / nchunks;
    uint8_t * d;
    CK(cudaMalloc(&d, n)); CK(cudaMemset(d, 1, n)); CK(cudaDeviceSynchronize());
    auto copy = [&](uint8_t * dst) {
        for (int i = 0; i < nchunks; i++) cudaMemcpy(dst + i * chunk, d + i * chunk, chunk, cudaMemcpyDeviceToHost);
    };
    { std::vector<uint8_t> w(64 << 20); cudaMemcpy(w.data(), d, 64 << 20, cudaMemcpyDeviceToHost); } // warm-up
    printf("state %zu MiB in %d copies of %.1f MiB\n", mib, nchunks, chunk / 1048576.0);
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_ms();
        { std::vector<uint8_t> v; v.resize(n); double t1 = now_ms(); copy(v.data()); double t2 = now_ms();
          printf("A today: vector resize (zero-fill) + copy : alloc+zero %7.1f ms | copy %7.1f ms | total %7.1f ms\n", t1 - t0, t2 - t1, t2 - t0); }
        t0 = now_ms();
        { std::unique_ptr<uint8_t[]> p(new uint8_t[n]); double t1 = now_ms(); copy(p.get()); double t2 = now_ms();
          printf("B fix 1: new[] without zero-fill + copy   : alloc      %7.1f ms | copy %7.1f ms | total %7.1f ms\n", t1 - t0, t2 - t1, t2 - t0); }
    }
    std::unique_ptr<uint8_t[]> r(new uint8_t[n]); memset(r.get(), 0, n);
    for (int rep = 0; rep < 3; rep++) { double t0 = now_ms(); copy(r.get()); printf("C fix 2: reused normal buffer             : copy %7.1f ms\n", now_ms() - t0); }
    uint8_t * h; double t0 = now_ms(); CK(cudaMallocHost(&h, n)); printf("D fix 3: one-time pinned allocation       : %7.1f ms\n", now_ms() - t0);
    for (int rep = 0; rep < 3; rep++) { double t1 = now_ms(); copy(h); printf("D fix 3: reused pinned buffer             : copy %7.1f ms\n", now_ms() - t1); }
    for (int rep = 0; rep < 3; rep++) { double t1 = now_ms(); memset(r.get(), 0, n); double t2 = now_ms(); copy(r.get());
        printf("E reused buffer, zero-filled again        : zero %7.1f ms | copy %7.1f ms | total %7.1f ms\n", t2 - t1, now_ms() - t2, now_ms() - t1); }
    const size_t n2 = (n + (2u << 20) - 1) & ~size_t((2u << 20) - 1);
    for (int zero = 1; zero >= 0; zero--) {
        for (int rep = 0; rep < 3; rep++) {
            double t1 = now_ms();
            uint8_t * m = (uint8_t *) mmap(nullptr, n2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (m == MAP_FAILED || madvise(m, n2, MADV_HUGEPAGE) != 0) { printf("mmap/madvise failed\n"); return 1; }
            if (zero) memset(m, 0, n);
            double t2 = now_ms(); copy(m); double t3 = now_ms();
            munmap(m, n2);
            printf("F fresh 2 MiB pages, %s: alloc%s %7.1f ms | copy %7.1f ms | total %7.1f ms\n",
                   zero ? "zero-filled + copy   " : "copy only            ", zero ? "+zero" : "     ", t2 - t1, t3 - t2, t3 - t1);
        }
    }
    cudaFreeHost(h); cudaFree(d);
    return 0;
}
