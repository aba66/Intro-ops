# 15分钟跑通 Gemma4 大模型：AMD GPU 部署学习笔记

这两天跟着 Datawhale 的 AMD 大模型实践任务，完整跑了一遍 Gemma4 的部署和对话流程。之前对“大模型部署”的理解还比较抽象，做完这次任务后发现，它其实可以拆成一条很清楚的链路：

```text
检查 GPU -> 下载模型 -> 安装推理框架 -> 启动服务 -> 连接对话
```

这篇记录一下我的实践过程和几个容易忽略的点。

## 1. 先确认 AMD GPU 能不能用

第一步不是急着下载模型，而是先检查云环境里的 AMD GPU 是否正常。

```bash
amd-smi
```

如果能看到显卡、显存、功耗等信息，说明 GPU 已经被系统识别。

然后再确认 PyTorch 能不能调用 ROCm 环境：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

这里有个小知识点：虽然命令里写的是 `torch.cuda`，但在 ROCm 版 PyTorch 里，它也可以对应 AMD GPU。重点看 `ROCm available` 是否为 `True`。

## 2. 用 ModelScope 下载 Gemma4

为了让依赖下载更稳定，先切换 pip 镜像：

```bash
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/
```

然后安装 ModelScope：

```bash
pip install modelscope
```

ModelScope 可以理解成国内常用的模型下载平台，用它拉大模型文件会更稳定。

下载本次任务使用的 Gemma4 模型：

```bash
modelscope download --model google/gemma-4-E4B-it --cache_dir "./models"
```

下载完成后可以检查模型目录：

```bash
ls -lh ./models/google/gemma-4-E4B-it/
```

重点看是否有 `model.safetensors` 之类的权重文件。下载过程可能要等几分钟，进度条短时间不动不一定是失败，别急着中断。

## 3. 安装 vLLM，把模型跑成服务

模型文件只是静态权重，不能直接聊天。真正让模型跑起来的是推理框架，这次用的是 vLLM。

```bash
uv pip uninstall torchvision
uv pip install vllm torchvision \
  --no-cache \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://wheels.vllm.ai/rocm/ \
  -U
```

这里最关键的是 ROCm 版本的 vLLM 安装源：

```text
https://wheels.vllm.ai/rocm/
```

AMD GPU 环境下，如果 vLLM 版本或依赖不匹配，后面很容易启动失败。

## 4. 启动 Gemma4 服务并对话

在第一个终端启动服务：

```bash
vllm serve ./models/google/gemma-4-E4B-it/ --served-model-name gemma-4-E4B-it
```

看到日志里出现下面这句，基本说明服务已经启动成功：

```text
Application startup complete.
```

注意：这个终端会一直被服务占用，不要关掉。

然后另开一个新终端，连接本地服务：

```bash
vllm chat --url http://localhost:8000/v1 --model gemma-4-E4B-it
```

可以输入：

```text
你是谁，你能做什么
```

如果模型能正常回复，就说明从模型下载到本地推理的流程已经跑通了。

## 5. 我遇到的几个关键理解点

第一，大模型部署不是一条命令解决的事情。它至少包含硬件检查、框架检查、模型下载、推理服务启动和客户端调用。

第二，模型文件本身不会“自己运行”。`model.safetensors` 只是保存权重，真正让它提供对话能力的是 vLLM 这样的推理框架。

第三，`vllm serve` 和 `vllm chat` 是服务端和客户端的关系。前者负责把模型加载起来并开放接口，后者只是去访问这个接口。以后做本地大模型应用、RAG 或 Agent，本质上也是这个思路。

第四，显存不够时可以先降低上下文长度：

```bash
vllm serve ./models/google/gemma-4-E4B-it/ --served-model-name gemma-4-E4B-it --max-model-len 8192
```

如果还是不够，可以继续降到 `4096`。这也让我意识到，部署大模型不只看参数量，还要看上下文长度、显存和推理框架配置。

## 6. 这次实践后的收获

以前看到“部署大模型”会觉得很重，但这次跑完后，整个过程变得具体了很多：

```text
GPU 是地基
模型权重是原料
vLLM 是发动机
API 接口是对外服务方式
```

只要按顺序验证，每一步都有明确的检查点。对初学者来说，这种最小闭环很重要，因为它能把“大模型”从概念变成一个真正能访问、能对话、能调试的服务。

下一步我想继续试试用 Python 直接请求 `http://localhost:8000/v1/chat/completions`，把本地 Gemma4 接到自己的小应用里。

#Datawhale #AMDev #Gemma4 #大模型部署 #AMD #ROCm #vLLM #开源大模型 #AI学习笔记
