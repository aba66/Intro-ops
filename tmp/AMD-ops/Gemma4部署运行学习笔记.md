# 15分钟部署并运行 Gemma4 大模型学习笔记

> 来源：Datawhale AI 学习中心「【Day1-2】15分钟部署&运行 Gemma4 大模型，撰写学习笔记」任务页面  
> 关键词：`#Datawhale` `#AMDev` `Gemma4` `AMD GPU` `ROCm` `vLLM` `ModelScope`

## 一、学习目标

本次任务的目标是把一个开源大模型从「模型文件」变成「可以对话的本地推理服务」。完整链路包括：

1. 在 AMD 云环境中确认 GPU 与 PyTorch ROCm 环境可用。
2. 使用国内镜像和 ModelScope 下载 Gemma4 模型权重。
3. 安装并使用 vLLM 启动模型服务。
4. 另开终端作为客户端，通过 OpenAI API 兼容接口与模型对话。
5. 理解大模型、参数、权重、推理、部署、ModelScope、vLLM 等基础概念。

我对这次任务的理解是：它不是单纯地复制命令，而是在实践「大模型部署」最小闭环。检查硬件、下载权重、启动推理框架、连接服务、完成对话，这几步正好对应了真实大模型应用上线前的基础流程。

## 二、环境检查

### 1. 检查 AMD GPU 是否可用

首先在 AMD 平台中打开终端，运行：

```bash
amd-smi
```

`amd-smi` 的作用类似于 NVIDIA 环境中的 `nvidia-smi`，用于查看 AMD GPU 的状态。只要能够看到 GPU 设备信息、显存、功耗等内容，就说明当前云环境中的显卡已经被系统识别。

### 2. 检查 PyTorch 是否能识别 ROCm GPU

接着运行：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

这里虽然代码里使用的是 `torch.cuda`，但在 ROCm 版本的 PyTorch 中，这套 CUDA 风格 API 也会映射到 AMD GPU。因此判断重点是：

- `ROCm available` 是否为 `True`。
- `Device` 是否能打印出显卡名称。

这一步很关键。如果 PyTorch 无法识别 GPU，后面即使成功下载模型，也无法正常用 GPU 加速推理。

## 三、下载 Gemma4 模型

Gemma4 是 Google DeepMind 推出的开源大模型家族。本次任务使用的是 `google/gemma-4-E4B-it`，也就是相对较小、适合单卡上手运行的 E4B 指令模型。

### 1. 切换 pip 镜像

为了提高国内环境下 Python 依赖安装速度，先把 pip 源切换到腾讯云镜像：

```bash
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/
```

这一步解决的是依赖下载速度和稳定性问题。大模型实验环境中经常要安装体积较大的包，如果下载源不稳定，会浪费大量时间。

### 2. 安装 ModelScope

```bash
pip install modelscope
```

ModelScope 是国内常用的开源模型社区，可以理解为模型下载站或模型应用商店。教程选择它的主要原因是模型文件在国内下载更稳定，适合教学和实验环境。

### 3. 下载 Gemma4 模型

```bash
modelscope download --model google/gemma-4-E4B-it --cache_dir "./models"
```

下载完成后，模型会保存在：

```text
./models/google/gemma-4-E4B-it/
```

可以用下面的命令查看目录内容：

```bash
ls -lh ./models/google/gemma-4-E4B-it/
```

重点关注是否存在 `model.safetensors` 等权重文件。教程中特别提醒：下载过程可能需要等待约 8 分钟，进度条短时间不动不一定代表失败，尤其是大文件下载时不要轻易中断。

## 四、安装并启动 vLLM

模型文件本身只是静态权重，不能直接对话。要让模型真正运行起来，需要一个推理框架负责加载模型、管理显存、接收请求并生成输出。这里使用的是 vLLM。

### 1. 安装 vLLM 与 torchvision

教程中给出的安装命令是：

```bash
uv pip uninstall torchvision # 经测试，在该云环境中，需卸载重新安装这个库才能正常使用
uv pip install vllm torchvision \
  --no-cache \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://wheels.vllm.ai/rocm/ \
  -U
```

这里有几个值得注意的点：

- 使用 `uv pip` 是为了更快地管理 Python 包。
- `--index-url` 指向阿里云 PyPI 镜像，用于加速通用依赖下载。
- `--extra-index-url https://wheels.vllm.ai/rocm/` 指向 vLLM 的 ROCm 轮子源，这是 AMD GPU 环境能正确安装 vLLM 的关键。
- 先卸载再安装 `torchvision` 是为了避免云环境中已有包版本不兼容。

### 2. 启动 vLLM 服务

```bash
vllm serve ./models/google/gemma-4-E4B-it/ --served-model-name gemma-4-E4B-it
```

这条命令做了几件事：

- 从 `./models/google/gemma-4-E4B-it/` 加载本地模型。
- 使用 `gemma-4-E4B-it` 作为服务暴露出来的模型名。
- 启动一个兼容 OpenAI API 的本地服务，默认地址通常是 `http://localhost:8000/v1`。

第一次启动可能较慢，因为需要加载模型、初始化显存、编译部分内核。只要日志还在输出，通常不需要中断。服务启动成功的重要标志是日志中出现：

```text
Application startup complete.
```

启动成功后，这个终端会被 vLLM 服务占用，不能关闭。它相当于服务端，持续等待客户端请求。

## 五、与 Gemma4 对话

因为第一个终端已经被服务占用，所以需要再打开一个新终端，作为客户端连接 vLLM 服务。

```bash
vllm chat --url http://localhost:8000/v1 --model gemma-4-E4B-it
```

进入聊天模式后，可以输入示例问题：

```text
你是谁，你能做什么
```

如果模型能够正常回复，说明部署流程已经完成：模型权重已下载、推理框架已启动、服务接口可访问、对话链路可用。

任务结束后，如果后续还要做微调或其他显存密集任务，需要在服务端终端使用 `Ctrl+C` 关闭 vLLM 服务，释放显存。

## 六、常见问题与排查思路

### 1. `vllm serve` 启动很慢

首次启动需要加载模型和编译内核，等待几分钟属于正常情况。判断是否应该继续等待，主要看日志是否仍然在输出。如果日志持续刷新，就不要轻易中断。

### 2. 显存不足

如果启动时报显存不足，可以降低最大上下文长度：

```bash
vllm serve ./models/google/gemma-4-E4B-it/ --served-model-name gemma-4-E4B-it --max-model-len 8192
```

如果仍然不够，可以继续降低到：

```text
4096
```

这说明上下文长度会直接影响显存占用。部署模型时，不是只看参数量，还要看上下文长度、batch、并发量等运行参数。

### 3. `modelscope download` 找不到

先检查 ModelScope 是否安装成功：

```bash
pip show modelscope
```

如果没有安装或版本异常，重新安装：

```bash
pip install -U modelscope
```

### 4. 聊天命令连接失败

优先检查三点：

1. 第一个终端中的 `vllm serve` 是否仍在运行。
2. 服务端日志中是否已经出现 `Application startup complete.`。
3. 客户端命令里的 `--url` 和 `--model` 是否与服务端一致。

也就是说，对话失败不一定是模型本身的问题，更多时候是服务还没启动完成、服务被关闭、端口不对或模型名不匹配。

## 七、核心概念整理

### 1. 大模型是什么

普通软件通常依赖程序员写死的规则，而大模型不是靠手写规则完成任务。它从海量文本中学习语言规律，本质上是在根据前文预测下一个最可能出现的词。

这个目标听起来简单，但要把下一个词预测得足够准确，模型就需要在训练中形成对语法、事实、常识、上下文关系和简单推理的统计性掌握。所以我们看到的大模型聊天、写作、代码生成等能力，本质上都建立在连续预测下一个 token 的机制上。

### 2. Gemma4 是什么

Gemma4 是 Google DeepMind 推出的开源大模型家族。本次任务页面中介绍它具有几个特点：

- 开源，权重可下载到本地运行。
- 商业友好，适合学习、实验和后续改造。
- 型号覆盖不同规模，本次使用的 E4B 比较适合单卡部署。
- 具备多语言、推理、代码、长上下文等能力。

我对它的定位理解是：Gemma4 适合作为开源大模型学习入口。它不像闭源模型那样只能通过远程 API 调用，而是可以真正下载到自己的环境里，观察模型文件、部署过程和推理服务形态。

### 3. 参数、权重与模型文件

- 参数：模型内部用于计算的大量数字。
- 权重：参数的另一种叫法。
- `B`：表示十亿参数，例如 4B 大约是 40 亿参数。
- 模型文件：例如 `model.safetensors`，里面保存的就是模型权重。

一句话总结：参数和权重就是模型能力的载体，下载模型本质上是在下载这些训练好的数字。

### 4. 推理与部署

- 推理：用训练好的模型生成结果，例如回答问题、写代码、总结文本。
- 部署：把模型运行成一个可访问的服务，让用户或程序可以请求它。

本次任务中的 `vllm serve` 就是部署动作，而 `vllm chat` 发起对话就是推理使用过程。

### 5. ModelScope 与 vLLM

ModelScope 解决的是「模型从哪里下载」的问题。它提供国内可用、速度相对稳定的模型下载渠道。

vLLM 解决的是「模型如何高效跑起来」的问题。它负责加载模型、管理显存、提供推理服务，并且暴露 OpenAI API 兼容接口，方便后续程序调用。

## 八、完整流程复盘

这次任务可以串成一条非常清晰的工程链路：

```text
检查 AMD GPU
  -> 检查 PyTorch ROCm
  -> 配置 pip 镜像
  -> 安装 ModelScope
  -> 下载 Gemma4 权重
  -> 安装 ROCm 版 vLLM
  -> 启动 vLLM 服务
  -> 新终端连接服务
  -> 输入问题验证对话
  -> 关闭服务释放显存
```

这条链路让我更直观地理解了大模型部署不是单点操作，而是硬件、驱动、Python 环境、模型权重、推理框架和服务接口之间的协同。

## 九、学习心得

通过这次任务，我最大的收获是把「运行一个大模型」拆成了可验证的步骤。以前容易把大模型部署理解成一个黑盒：下载一个模型，然后运行某条命令。但实际上每一步都有明确目的：

- `amd-smi` 验证硬件层是否可用。
- `torch.cuda.is_available()` 验证深度学习框架是否能使用 AMD GPU。
- ModelScope 负责把远程模型权重拉到本地。
- vLLM 负责把静态权重加载成可访问的推理服务。
- `vllm chat` 则是客户端调用服务完成真实推理。

我也意识到，部署大模型时最常见的问题并不一定来自模型能力，而是来自工程环境：显卡是否识别、ROCm 依赖是否匹配、vLLM 轮子是否安装正确、模型路径是否正确、服务是否真正启动完成、显存是否够用。这些问题都需要用日志和检查命令逐层定位。

最后，这个任务也让我理解了「服务端 + 客户端」模式在大模型应用中的意义。`vllm serve` 启动的是后端服务，`vllm chat` 只是一个访问它的客户端。以后无论是写 Web 应用、Agent、RAG 系统，还是用 OpenAI API 兼容接口调用本地模型，本质上都可以复用这个思路：先让模型稳定作为服务运行，再让业务程序通过接口访问它。

## 十、后续计划

后续如果继续深入，我会重点关注三件事：

1. 学习如何调节 `--max-model-len`、并发数、显存利用率等 vLLM 参数。
2. 尝试用 Python 请求 `http://localhost:8000/v1/chat/completions`，把本地 Gemma4 接入自己的应用。
3. 在理解部署流程后，继续学习模型微调，把开源模型从「能运行」推进到「更适合特定任务」。
