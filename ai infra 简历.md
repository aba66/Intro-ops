# AI Infra 方向简历项目经历

> 建议按实际完成内容删改，简历里优先写自己能讲清楚架构、调试过程和性能结果的部分。

## 项目名称

轻量级 GPU 算子运行时与 AI Infra 算子开发框架

## 项目描述

基于 C++/CUDA/Python 构建一个用于学习 AI Infra 算子开发链路的轻量级 Operator Runtime。项目采用后端无关的 C ABI 组织算子生命周期，并通过 Python binding 暴露给上层调用；底层实现 NVIDIA CUDA 算子，配套 CMake/Ninja 构建、pytest 正确性测试和 benchmark，用于理解深度学习框架中算子注册、参数校验、运行时调度、kernel 执行和性能评估的完整流程。

## 简历 Bullet

- 设计并实现轻量级 GPU Operator Runtime，采用 C ABI 抽象算子生命周期，包括 descriptor 创建、workspace 查询、算子执行和资源销毁，降低上层 Python 调用与底层 CUDA 实现之间的耦合。
- 实现 Python binding 与动态库加载机制，通过 ctypes 绑定 C API，并将 PyTorch Tensor 的 data pointer、shape、stride、dtype 等信息转换为运行时 TensorView。
- 实现 NVIDIA 后端基础算子，包括 copy、vector_add、reduce_sum、softmax、relu，覆盖 FP32/FP16、contiguous tensor、row-wise reduction 和 elementwise 等常见算子模式。
- 使用 CUDA grid-stride loop、shared memory reduction、numerically stable softmax、half precision arithmetic 等方式完成 kernel 编写，并通过 PyTorch 结果进行正确性校验。
- 集成 CUTLASS/CuTe，用于实现向量化 copy kernel，处理对齐访问、尾部 predicate、模板类型推导和 CUDA/CUTLASS 版本兼容等工程问题。
- 搭建 CMake/Ninja/Conda 构建流程，支持 NVIDIA 后端自动构建、CUTLASS 拉取、动态库加载路径配置和本地环境复现。
- 建立算子级测试与 benchmark 流程，使用 pytest 覆盖 API contract 和 correctness case，并基于 PyTorch profiler/CUDA event 统计 runtime latency，与 PyTorch baseline 对比性能。
- 熟悉 AI Infra 算子开发完整链路，包括算子接口设计、Tensor 元信息抽象、运行时 ABI、CUDA stream 管理、workspace 管理、kernel 调试、性能 profiling 和端到端验证。

## AI Infra 强调版

如果简历篇幅有限，可以压缩成下面这一版：

轻量级 GPU 算子运行时与 AI Infra 算子开发框架：基于 C++/CUDA/Python 实现后端无关的算子运行时，通过 C ABI 管理 descriptor、workspace、execute、destroy 等生命周期，并使用 ctypes 暴露 Python 调用接口；实现 copy、vector_add、reduce_sum、softmax、relu 等 NVIDIA CUDA 算子，覆盖 elementwise、reduction、softmax 等典型模式；集成 CUTLASS/CuTe 向量化 copy，并搭建 CMake/Ninja 构建、pytest 正确性测试和 benchmark 流程，理解深度学习框架中算子从 Python 到 CUDA kernel 的执行链路。

## 后续新增算子后可增强

- 新增 LayerNorm、RMSNorm、RoPE、MatMul/GEMV、TopK、Quantize/Dequantize 等更贴近大模型推理的算子。
- 为每个新增算子补充 descriptor、Python API、correctness tests、benchmark cases 和 examples，体现完整算子接入能力。
- 对比 naive CUDA、shared memory 优化、向量化访存、CUTLASS/CuTe 或 warp-level primitive 的性能差异。
- 记录关键指标，例如 latency、memory bandwidth、speedup、误差范围、支持 dtype 和输入规模，面试时可以用数据讲清楚优化收益。
- 如果继续扩展框架，可以加入算子注册表、后端选择、fallback、workspace planner、graph-level fusion 或多后端适配能力。
