"""Run metadata, GPU sampling, CPU pinning and server-log facts, shared by the benchmark scripts.

Every server start (grid.start) writes <name>.meta.json:
  build: llama-server --version, flags from the CMake cache, sha256 of the binary and its shared libraries
  model: path, size, mtime and a hash of its first and last 16 MiB
  run:   full command line (with the taskset prefix when pinned), LLAMA_/GGML_/CUDA_/NSYS_ environment variables
  host:  CPU governor and frequency, transparent huge pages, CPU pinning
  gpu:   driver, persistence mode, clocks, temperature, power and other processes on the GPU, before and after the run
While the server runs, nvidia-smi samples the GPU every GPU_SAMPLE_MS ms (default 500, 0 = off) into <name>.gpu.csv;
grid.stop adds a summary of the samples to the meta file: SM clock range, temperature, power and how often each
clock-limit reason (power cap, thermal, ...) was active.

CPU pinning keeps the load generator off the cores the server uses (taskset-style lists; unset = no pinning):
  SERVER_CPUS=0-23 CLIENT_CPUS=24-31
"""
import collections
import glob
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

GPU_STATE_FIELDS = ("name,driver_version,persistence_mode,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,"
                    "power.draw,power.limit,pstate,memory.used,utilization.gpu")
ENV_PREFIXES = ("LLAMA_", "GGML_", "CUDA_", "NSYS_", "SERVER_CPUS", "CLIENT_CPUS", "GPU_SAMPLE_MS")
# bits of clocks_event_reasons.active (nvidia-smi --help-query-gpu)
REASONS = {0x1: "gpu idle", 0x2: "applications clocks", 0x4: "sw power cap", 0x8: "hw slowdown", 0x20: "sw thermal",
           0x40: "hw thermal", 0x80: "hw power brake", 0x100: "display clocks"}

_hash_cache = {}
_reason_field = None


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"error: {e!r}"


def cpu_list(spec):
    # taskset-style list, e.g. "0-3,8" -> {0, 1, 2, 3, 8}
    cpus = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def pin_client():
    spec = os.environ.get("CLIENT_CPUS")
    if spec:
        os.sched_setaffinity(0, cpu_list(spec))


def server_cmd(cmd):
    spec = os.environ.get("SERVER_CPUS")
    if not spec:
        return cmd
    client = os.environ.get("CLIENT_CPUS")
    if client and cpu_list(spec) & cpu_list(client):
        print(f"warning: SERVER_CPUS={spec} and CLIENT_CPUS={client} overlap", file=sys.stderr, flush=True)
    return ["taskset", "-c", spec] + cmd


def _sha256(path):
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def build_info(bin_dir):
    # the CMake cache is in the build directory, one level above bin/
    exe = os.path.join(bin_dir, "llama-server")
    info = {"bin_dir": os.path.abspath(bin_dir), "version": _run([exe, "--version"])[-2000:]}
    files = sorted({os.path.realpath(p) for p in [exe] + glob.glob(os.path.join(bin_dir, "*.so*")) if os.path.isfile(p)})
    info["sha256"] = {os.path.basename(p): _sha256(p) for p in files}
    cache = os.path.join(os.path.dirname(os.path.abspath(bin_dir)), "CMakeCache.txt")
    keys = ("CMAKE_BUILD_TYPE", "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_CUDA_FLAGS", "CMAKE_CUDA_ARCHITECTURES",
            "CMAKE_CUDA_COMPILER", "CMAKE_HOME_DIRECTORY", "CUDAToolkit_ROOT", "GGML_CUDA", "GGML_NATIVE", "BUILD_SHARED_LIBS")
    if os.path.exists(cache):
        for line in open(cache, errors="replace"):
            k, _, v = line.strip().partition("=")
            if k.split(":")[0] in keys:
                info[k.split(":")[0]] = v
    flags = " ".join(info.get(k, "") for k in ("CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS"))
    info["profiling_build"] = "-fno-omit-frame-pointer" in flags or "DEBUG_TIMINGS" in flags
    return info


def model_info(path):
    # quick identity: size, mtime and a hash of the first and last 16 MiB (hashing the whole file takes minutes)
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(16 << 20))
        f.seek(max(0, st.st_size - (16 << 20)))
        h.update(f.read(16 << 20))
    return {"path": os.path.abspath(path), "size": st.st_size, "mtime": st.st_mtime, "sha256_head_tail_16MiB": h.hexdigest()}


def host_info():
    govs = sorted({open(p).read().strip() for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")})
    mhz = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
    thp = "/sys/kernel/mm/transparent_hugepage/enabled"
    return {
        "host": platform.node(), "kernel": platform.release(), "cpu_count": os.cpu_count(), "cpu_governors": govs,
        "cpu_mhz_mean": statistics.mean(mhz) if mhz else None, "thp": open(thp).read().strip() if os.path.exists(thp) else None,
        "server_cpus": os.environ.get("SERVER_CPUS"), "client_cpus": os.environ.get("CLIENT_CPUS"),
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }


def _gpu_select():
    # the GPU the server sees first
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return ["-i", dev] if dev else []


def gpu_info():
    return {
        "fields": GPU_STATE_FIELDS,
        "values": _run(["nvidia-smi"] + _gpu_select() + [f"--query-gpu={GPU_STATE_FIELDS}", "--format=csv,noheader"]),
        "apps": _run(["nvidia-smi"] + _gpu_select() + ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
    }


def gpu_apps():
    # compute processes on the GPU, e.g. a benchmark started from another terminal
    out = _run(["nvidia-smi"] + _gpu_select() + ["--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if out.startswith("error:"):
        return []  # no nvidia-smi: nothing to check
    return [l for l in out.splitlines() if l.strip() and not l.startswith("No running")]


def write_meta(path, cmd, bin_dir):
    model = cmd[cmd.index("-m") + 1] if "-m" in cmd else None
    meta = {
        "time": time.time(), "cmd": cmd, "build": build_info(bin_dir),
        "model": model_info(model) if model and os.path.exists(model) else model,
        "env": {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES)},
        "host": host_info(), "gpu_before": gpu_info(),
    }
    json.dump(meta, open(path, "w"), indent=1)
    if meta["build"]["profiling_build"]:
        print(f"note: {bin_dir} is a profiling build (frame pointers or DEBUG_TIMINGS); take timing numbers from a plain Release build",
              file=sys.stderr, flush=True)
    return meta


def _reasons_field():
    # renamed from clocks_throttle_reasons in newer drivers
    global _reason_field
    if _reason_field is None:
        r = subprocess.run(["nvidia-smi"] + _gpu_select() + ["--query-gpu=clocks_event_reasons.active", "--format=csv,noheader"],
                           capture_output=True, text=True)
        _reason_field = "clocks_event_reasons.active" if r.returncode == 0 else "clocks_throttle_reasons.active"
    return _reason_field


def summarize_gpu_csv(path):
    # columns: timestamp, SM clock, memory clock, temperature, power, utilization, clock-limit reasons (bit mask)
    rows = []
    if os.path.exists(path):
        for line in open(path):
            f = [x.strip() for x in line.split(",")]
            if len(f) < 7:
                continue
            try:
                rows.append((float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]), int(f[6], 16)))
            except ValueError:
                continue
    if not rows:
        return {"samples": 0}
    sm, mem, temp, power, util, reasons = zip(*rows)
    n_busy = sum(1 for u in util if u > 0)
    busy = [r for r in rows if r[4] > 0] or rows
    return {
        "samples": len(rows), "busy_samples": n_busy,
        # clocks while the GPU was busy (all samples if it never was); idle samples sit at the idle clock
        "sm_clock_mhz_busy_min": min(r[0] for r in busy), "sm_clock_mhz_busy_median": statistics.median(r[0] for r in busy),
        "sm_clock_mhz_max": max(sm), "mem_clock_mhz_min": min(r[1] for r in busy),
        "temp_c_max": max(temp), "power_w_mean": statistics.mean(power), "power_w_max": max(power), "util_pct_mean": statistics.mean(util),
        "clock_limit_share": {name: round(sum(1 for m in reasons if m & bit) / len(rows), 3)
                              for bit, name in REASONS.items() if bit != 0x1 and any(m & bit for m in reasons)},
    }


class GpuMonitor:
    # samples the GPU into a CSV until stop()
    def __init__(self, path):
        self.path = path
        ms = int(os.environ.get("GPU_SAMPLE_MS", "500"))
        self.proc = None
        if ms > 0:
            fields = f"timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,{_reasons_field()}"
            self.proc = subprocess.Popen(["nvidia-smi"] + _gpu_select() + [f"--query-gpu={fields}", "--format=csv,noheader,nounits",
                                          "-lms", str(ms), "-f", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.proc is None:
            return {"samples": 0}
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return summarize_gpu_csv(self.path)


def finish_meta(path, monitor):
    meta = json.load(open(path))
    meta["gpu_during_run"] = monitor.stop()
    meta["gpu_after"] = gpu_info()
    meta["host_after"] = {"cpu_mhz_mean": host_info()["cpu_mhz_mean"], "loadavg": open("/proc/loadavg").read().split()[:3]}
    json.dump(meta, open(path, "w"), indent=1)
    return meta


def log_stats(path):
    # facts from a server log: slot setup, prompt-cache evictions, per-request decode speed and, when the build defines
    # DEBUG_TIMINGS in server-context.cpp, the step-phase averages (that build synchronizes after every decode, so its
    # timing numbers are not comparable with a normal build)
    text = open(path, errors="replace").read() if os.path.exists(path) else ""
    out = {"evictions": text.count("making room for prompt cache entry")}
    m = re.search(r"n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(\w+)'", text)
    if m:
        out.update({"n_slots": int(m.group(1)), "n_ctx_slot": int(m.group(2)), "kv_unified": m.group(3)})
    mspt = [float(x) for x in re.findall(r"\|\s+eval time =\s+[\d.]+ ms /\s+\d+ tokens \(\s*([\d.]+) ms per token", text)]
    out["server_ms_per_token_median_last"] = sorted(mspt[-16:])[len(mspt[-16:]) // 2] if mspt else None
    # the server prints these every 5 s as averages since it started; the last print wins
    dbg = collections.OrderedDict()
    for k, v in re.findall(r"avg (t_\w+)\s*= ([\d.]+) ms", text):
        dbg[k] = float(v)
    if dbg:
        out["debug_timings_avg_ms"] = dict(dbg)
    return out
