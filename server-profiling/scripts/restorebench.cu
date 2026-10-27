// copy one slot-sized KV state host -> GPU the way llama_state_seq_set_data_ext() restores it: one
// ggml_backend_tensor_set() (cudaMemcpyAsync + stream sync) per destination run of cells, per layer, for K and V,
// from a pageable buffer that was filled before (a snapshot), or a pinned one for comparison
// usage: restorebench [state MiB = 1560] [layers x 2 = 96] [run KiB for the fragmented case = 1]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? atoll(argv[1]) : 1560;
    const int nchunks = argc > 2 ? atoi(argv[2]) : 96; // llama.cpp copies K and V per layer: 48 x 2
    const size_t run = (argc > 3 ? atoll(argv[3]) : 1) << 10;
    const size_t n = mib << 20, chunk = n / nchunks;
    uint8_t * d;
    CK(cudaMalloc(&d, n));
    std::vector<uint8_t> v(n, 1); // a snapshot: pages already touched
    uint8_t * p;
    CK(cudaMallocHost(&p, n)); memset(p, 1, n);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    auto set = [&](const uint8_t * src, size_t off, size_t len) {
        cudaMemcpyAsync(d + off, src + off, len, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
    };
    set(v.data(), 0, 64 << 20); // warm-up
    printf("state %zu MiB: %d copies of %.1f MiB (contiguous), or %zu copies of %zu KiB (one per run)\n",
           mib, nchunks, chunk / 1048576.0, n / run, run >> 10);
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_ms();
        for (int i = 0; i < nchunks; i++) set(v.data(), i * chunk, chunk);
        double t1 = now_ms();
        for (int i = 0; i < nchunks; i++) set(p, i * chunk, chunk);
        double t2 = now_ms();
        for (size_t off = 0; off + run <= n; off += run) set(v.data(), off, run);
        double t3 = now_ms();
        for (size_t off = 0; off + run <= n; off += run) set(p, off, run);
        double t4 = now_ms();
        printf("contiguous: pageable %7.1f ms (%5.2f GB/s), pinned %7.1f ms (%5.2f GB/s) | "
               "per run: pageable %7.1f ms (%5.2f us/copy), pinned %7.1f ms (%5.2f us/copy)\n",
               t1 - t0, n / (t1 - t0) / 1e6, t2 - t1, n / (t2 - t1) / 1e6,
               t3 - t2, (t3 - t2) * 1e3 / (n / run), t4 - t3, (t4 - t3) * 1e3 / (n / run));
    }
    cudaFreeHost(p);
    cudaFree(d);
    return 0;
}
