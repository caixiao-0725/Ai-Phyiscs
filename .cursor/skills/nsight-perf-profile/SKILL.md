---
name: nsight-perf-profile
description: >-
  使用 NVIDIA Nsight Systems 对 ChysX 可执行文件进行性能分析。分析各部分用时占比，
  识别 CPU/GPU 热点和数据传输瓶颈，提出针对性优化建议。当用户提到性能慢、需要
  profiling、优化、用时分析时使用此 skill。
---

# Nsight Systems 性能分析

## 流程

### 1. 采集 trace

```bash
nsys profile --stats=true --force-overwrite=true -o <output_name> <exe> <args>
```

- 用 `--stats=true` 在 CLI 直接输出统计摘要
- 用 `--trace=cuda,nvtx,osrt` 捕获 CUDA API、内核、OS runtime
- 若只需 CPU 分析可加 `--trace=osrt`
- 输出 `.nsys-rep` 文件可在 Nsight Systems GUI 打开

### 2. 导出统计报告

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum <output_name>.nsys-rep
```

关键报告类型：
| 报告 | 用途 |
|------|------|
| `cuda_gpu_kern_sum` | GPU kernel 总耗时排名 |
| `cuda_api_sum` | CUDA API 调用（含 memcpy）耗时排名 |
| `cuda_gpu_mem_size_sum` | 显存传输量统计 |
| `cuda_gpu_mem_time_sum` | 显存传输耗时统计 |
| `osrt_sum` | OS runtime（malloc, I/O 等）耗时 |

### 3. 分析要点

1. **识别时间大头**：按 kernel/API 调用的 Total Time 降序排列，前 3-5 项通常占 80%+ 时间
2. **CPU-GPU 传输**：检查 `cudaMemcpy` (HtoD / DtoH) 次数和总耗时，高频小量传输是常见瓶颈
3. **同步阻塞**：`cudaDeviceSynchronize` / `cudaStreamSynchronize` 调用次数和等待时间
4. **kernel 占用率**：kernel 执行时间 vs GPU idle 时间的比值
5. **CPU 端瓶颈**：若 CPU 时间远大于 GPU 时间，说明瓶颈在 host 端（如暴力搜索、序列化操作）

### 4. 优化建议模板

针对每个瓶颈输出：
- **瓶颈**：具体函数/操作名 + 占总时间百分比
- **原因**：为什么慢（算法复杂度、传输频次、同步等）
- **建议**：具体优化方案 + 预期收益

### 5. CPU-GPU 传输优化检查清单

- 使用 pinned memory (`cudaMallocHost`) 替代 pageable memory
- 合并多次小传输为一次大传输
- 使用 `cudaMemcpyAsync` + stream 重叠计算和传输
- 避免不必要的 DtoH 回传（如只在 CPU 读取诊断值时才 sync）
- 对频繁使用的小数据考虑在 GPU 上保留（避免来回传输）

### 6. 用 nsys 获取函数级耗时

```bash
nsys profile --stats=true --trace=cuda,nvtx,osrt --sample=process-tree \
    -o <output_name> <exe> <args>
```

`--sample=process-tree` 开启 CPU 采样，可得到每个函数的 CPU 耗时。结合 `--trace=cuda` 可同时获取 GPU kernel 时间。`nsys stats` 的 `--report` 支持 `cuda_gpu_kern_sum` 查看每个 kernel 的总/平均/最大耗时排名。

### 7. CPU 计时测量 CUDA 操作的注意事项

用 `std::chrono` 或 `cudaEvent` 在 CPU 端测量 CUDA 操作耗时时，**必须先同步**：

```cpp
cudaDeviceSynchronize();                 // 等所有 GPU 操作完成
auto t0 = std::chrono::steady_clock::now();
launch_my_kernel<<<grid, block>>>(...);
cudaDeviceSynchronize();                 // 等 kernel 完成
auto t1 = std::chrono::steady_clock::now();
```

不加 `cudaDeviceSynchronize()` 的话，`std::chrono` 只测到了 kernel launch 的开销（微秒级），不是 kernel 实际执行时间。对于多个连续 kernel，在**首个 kernel 前**和**最后一个 kernel 后**各做一次 sync 即可，避免过多 sync 导致 pipeline stall。
