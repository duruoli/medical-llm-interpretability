# Lambda instance 设置手册

每次开 Lambda GPU 机器时照着做。路径和命令都是 2026-11-10/11 那次实际用过的。

---

## 速查：每次都要做的 6 步

1. 控制台 **Launch instance**：选机型、SSH key `Duruo-1`、**挂上 filesystem `medllm`**（只能在启动时挂）。
2. 复制新 instance 的 IP，改本机 `~/.ssh/config` 里 `Host lambda-gpu` 的 `HostName`。
3. 本机测试：`ssh lambda-gpu nvidia-smi`。
4. 同步代码（和需要的数据）到 instance，见第 4 节。
5. 在 `tmux` 里跑任务，见第 5 节。
6. 拷回结果，然后到控制台 **Terminate**，见第 6 节。

> ⚠️ Lambda 没有暂停。从启动到 **Terminate** 一直按分钟计费，不管 GPU 有没有在用。不要用 `sudo shutdown`：机器会进入告警状态，**继续收费**。

---

## 1. 一次性准备（已做过，换电脑时才需要）

```bash
code --install-extension ms-vscode-remote.remote-ssh   # VS Code 远程连接插件
pbcopy < ~/.ssh/id_rsa.pub                              # 复制公钥，粘贴到控制台 SSH keys 页面
```

- 控制台里的 key 叫 `Duruo-1`，对应本机 `~/.ssh/id_rsa`。
- `~/.ssh/id_rsa`（不带 `.pub`）是私钥，不要发给任何人。

本机 `~/.ssh/config` 里有这一段：

```
Host lambda-gpu
    HostName <instance 的 IP>
    User ubuntu
    IdentityFile ~/.ssh/id_rsa
```

---

## 2. 启动 instance

**机型**（价格以控制台为准）：

| 机型 | GPU 显存 | CPU | 内存 | 适合 |
|---|---|---|---|---|
| **H100 PCIe** | 80 GB | 26 | 225 GB | Qwen3-14B 的 AtP / patching / Microscope 收集（实测峰值约 45 GB）。**默认选这个** |
| A100（单卡） | 40 GB | 30 | 225 GB | 只做推理、小 batch。注意 Lambda 单卡 A100 只有 40 GB |
| A10 | 24 GB | 30 | 226 GB | 放不下 14B 的 bf16 权重（29.5 GB），别选 |

**Filesystem `medllm`**：
- 启动向导里勾选挂载。**启动后无法再挂**，忘了就只能 Terminate 重开。
- 它绑定地区，所以必须选**和 filesystem 同一个地区**的机器。
- 挂载位置：`/lambda/nfs/medllm`，家目录里也有快捷方式 `~/medllm`。
- instance 删除后它还在，并且一直按 GB 每月收费。

---

## 3. 连接

**改 IP**：打开本机 `~/.ssh/config`，把 `HostName` 换成新 IP。每个新 instance 的 IP 都不一样。

**测试**（本机终端）：

```bash
ssh lambda-gpu nvidia-smi
```

**VS Code**：Cmd+Shift+P → `Remote-SSH: Connect to Host` → `lambda-gpu` → 系统选 Linux → 打开文件夹 `/lambda/nfs/medllm`。

**让 Claude 直接操作**：告诉 Claude「instance 已开，IP 是 …」。它可以从本机用 `ssh lambda-gpu` 跑命令，用 `rsync` 传文件。

---

## 4. 环境与文件

### 4a. filesystem 里已有的东西（之前搭好，直接用）

| 路径（都在 `/lambda/nfs/medllm/` 下） | 内容 |
|---|---|
| `venv/` | 本项目的 Python 环境（transformers 5.x），给 `neuron_posneg.py` 等用 |
| `hf/` | Hugging Face 缓存，里面有 Qwen3-14B 权重（约 30 GB） |
| `medical-llm-interpretability/` | 代码副本，包括 `data/` 和 `results/` |
| `Neuron-Microscope/` | 工具代码（commit b5d63b7，已打补丁） |
| `nm-venv/` | 工具专用环境（transformers 4.57.6，torch 2.7.1） |
| `corpus/pmc_patients.jsonl` | Microscope 语料（PMC-Patients V2，公开病例报告） |

先确认它们都还在：

```bash
ssh lambda-gpu 'cd /lambda/nfs/medllm && ls; ls hf/hub 2>/dev/null'
```

### 4b. 同步代码（每次改过代码后）

代码不走 GitHub clone，而是从本机用 rsync 只传 git 跟踪的文件。在本机仓库根目录运行：

```bash
git ls-files > /tmp/tracked.txt
rsync -az --files-from=/tmp/tracked.txt ./ lambda-gpu:/lambda/nfs/medllm/medical-llm-interpretability/
```

只改了一个文件时：

```bash
scp neuron_posneg.py lambda-gpu:/lambda/nfs/medllm/medical-llm-interpretability/
```

### 4c. 数据（MIMIC，只走 rsync/scp，绝不进 GitHub）

```bash
rsync -az data/posneg_pool_v2.jsonl lambda-gpu:/lambda/nfs/medllm/medical-llm-interpretability/data/
```

### 4d. filesystem 是空的时（新地区或已删除）：从头搭

在 instance 上运行，预计 30–40 分钟，主要是下载：

```bash
cd /lambda/nfs/medllm
python3 -m venv venv && source venv/bin/activate
# 先用 4b 把代码 rsync 过来，再：
pip install -r medical-llm-interpretability/requirements.txt
export HF_HOME=/lambda/nfs/medllm/hf
hf download Qwen/Qwen3-14B
```

Microscope 环境要单独建，不和上面混用：

```bash
cd /lambda/nfs/medllm
git clone https://github.com/RUC-TrustworthyAI/Neuron-Microscope.git
cd Neuron-Microscope && git checkout b5d63b7
python3 -m venv ../nm-venv && source ../nm-venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r language/requirements/gpu.txt
```

然后重新打上工具补丁。补丁内容见 `results/neuron/Qwen3-14B/HANDOFF_quest.md` 的 “Tool patches to re-apply” 一节，最好让 Claude 来做。

### 4e. OpenRouter key（只放在 instance 上，不进任何文件或 git）

在 instance 的终端里**自己输入**，屏幕不会显示：

```bash
read -rs KEY && printf '%s' "$KEY" > ~/.openrouter_key && chmod 600 ~/.openrouter_key && unset KEY
```

运行工具时设置 `export NEURON_PAPER_API_KEY_FILE=~/.openrouter_key`。

注意：`~` 在系统盘上，Terminate 后会消失，每次新机器都要重输一次。

---

## 5. 运行

长任务一定放进 `tmux`，这样关掉 VS Code 或断网都不会中断。

```bash
tmux new -s neuron                     # 新开会话；Ctrl+B 再按 D 可以挂到后台
cd /lambda/nfs/medllm/medical-llm-interpretability
source ../venv/bin/activate
export HF_HOME=/lambda/nfs/medllm/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u neuron_posneg.py --model Qwen/Qwen3-14B
```

- 回到会话：`tmux attach -t neuron`。
- 查看 GPU：`nvidia-smi`。

Microscope 的运行方式不同：用 `nm-venv` 环境，另加 `NEURON_MICROSCOPE_GPU_WORKER=1`，命令见 HANDOFF 文件。

---

## 6. 结束（每次都做）

**1. 拷回结果**（本机运行，跳过很大的 `.npy`）：

```bash
rsync -az --exclude '*.npy' \
  lambda-gpu:/lambda/nfs/medllm/medical-llm-interpretability/results/<实验目录> results/
```

**2. 确认没有任务还在跑**：

```bash
ssh lambda-gpu 'tmux ls; nvidia-smi --query-gpu=memory.used --format=csv'
```

**3. 控制台 → Instances → Terminate。** 系统盘上的东西（`~` 下的 key、`/tmp`）会全部消失，只有 filesystem `medllm` 保留。

**4.（可选）彻底结束项目时删除 filesystem。** 它一直计费，而且里面有 MIMIC 数据。删除前要先 Terminate 所有挂着它的 instance。

---

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `Permission denied (publickey)` | 启动时没选 `Duruo-1` key，只能 Terminate 重开并选对 key |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | 新机器碰巧用了旧 IP。运行 `ssh-keygen -R <IP>` 后重连 |
| `/lambda/nfs/medllm` 不存在 | 启动时没挂 filesystem，或地区不同。只能 Terminate 重开 |
| `venv` 里 python 报错找不到 | 新机器的系统镜像换了 Python 版本，按 4d 重建 venv（权重不用重下） |
| CUDA out of memory | 换 H100 80 GB，或调小 batch |
| 打分随 batch 大小变化（最多 0.5） | bf16 的数值误差，patching 和 steering 的测量用 float32 |
