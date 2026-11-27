# 安装与下载

| 用途 | 获取方式 | 运行位置 |
|---|---|---|
| 采集模型，或从 trace 生成报告 | [PyPI 的 Python 包](https://pypi.org/project/serveutil/0.1.1/)，运行 `pip install serveutil` | 在采集机器的推理环境中测量，在普通电脑上也能生成报告 |
| 在独立桌面窗口分析已有报告 | v0.1.1 桌面查看器：[Windows x86-64](https://github.com/HaoyangPing0324/ServeUtil/releases/download/v0.1.1/ServeUtil-Windows-AMD64.zip) · [macOS Apple silicon](https://github.com/HaoyangPing0324/ServeUtil/releases/download/v0.1.1/ServeUtil-Darwin-arm64.tar.gz) · [Linux x86-64](https://github.com/HaoyangPing0324/ServeUtil/releases/download/v0.1.1/ServeUtil-Linux-x86_64.tar.gz) | 下载到查看报告的电脑，解压启动，内含 Python 运行时 |
| 在浏览器中查看已有报告 | [打开网页版查看器](https://haoyangping0324.github.io/ServeUtil/viewer.html?lang=zh) | 在当前浏览器处理文件，免安装 |

**Package** 是由包管理器安装的一份软件。ServeUtil 的 Python 包提供 `serveutil` 命令和 `serveutil.profile` 接口。**Release** 是项目为某个标签发布的版本页，集中提供发行说明、源码和各平台附件。同一版本可以同时在 PyPI 提供采集包、在 GitHub Release 提供桌面查看器。

## 网页版查看器

点击导航栏的「打开查看器」，拖入 trace 或选择整个采集文件夹。文件在浏览器本地处理。也可以直接打开教程中的真实样例。

## 桌面客户端

上表提供 v0.1.1 的固定版本。下方另列与当前网站提交对应的已测试构建。下载后完整解压。Windows 打开 **ServeUtil.exe**，macOS 打开 **ServeUtil.app**，Linux 运行解压目录内的 **ServeUtil**。桌面窗口可离线读取 JSON、gzip、ServeUtil 报告和原生 Perfetto trace。

## Python 命令行与模型采集

使用 Python 3.10 或更高版本。已发布版本在 PyPI 上的安装方式为：

```bash
pip install serveutil
serveutil --help
```

如需使用 GitHub 仓库的最新源码，运行以下命令。

```bash
python -m pip install "serveutil @ git+https://github.com/HaoyangPing0324/ServeUtil.git"
```

录制时使用模型现有的 PyTorch 环境。兼容 Linux 设备需要独立 CUPTI HBM 采样时安装 `serveutil[pm]`。从 Python 环境启动桌面窗口时安装 `serveutil[desktop]`。[第一次采集](start.md)

## v0.1.1 版本依赖与验证范围

| 用法 | 安装条件与验证版本 |
|---|---|
| 基础 CLI 与报告 | Python ≥3.10。pip 自动安装 `ijson>=3.2,<4`、`prometheus-client>=0.17,<1`、`nvidia-ml-py>=12.550.52`。分析已有报告时可以使用无 GPU 的电脑。 |
| PyTorch 进程内采集 | 使用模型环境自带的 PyTorch。CPU 采集验证了 2.10.0、2.15.0；H20 上验证了 2.13.0+cu130、2.15.0+cu132。GPU 采集还需与该 PyTorch 构建匹配的驱动。 |
| 服务端采集 | 只需安装实际运行的框架并开启其 PyTorch profiler。同机测过 vLLM 0.29.0、SGLang 0.5.21；双物理机测过 vLLM 0.32.0、SGLang 0.5.22。多节点的各采集进程使用同一 ServeUtil 版本。 |
| 可选依赖 | Linux 的独立 CUPTI PM 采样使用 `serveutil[pm]`，安装 `cupti-python>=13.4,<14`，并需设备、驱动与计数器权限支持。从 Python 启动桌面界面使用 `serveutil[desktop]`，安装 `pywebview>=6,<7`（Linux 使用 Qt 后端）。导入 llmperfscope 采样可安装 `serveutil[llmperfscope]`，范围为 `>=0.2,<1`。 |
| 查看器 | 网页端支持现代浏览器，已用 Chromium 验证。桌面发行包按上表架构构建，Linux 在 Ubuntu 24.04 验证。完整 trace 视图随附 Perfetto UI 和 Trace Processor v59.1。 |

框架版本是实测组合。复现实验可固定安装 `pip install serveutil==0.1.1`，再按模型服务需要选择 PyTorch、CUDA 和推理框架。[模型实测结果](validation.md)

## 版本记录

[GitHub Releases](https://github.com/HaoyangPing0324/ServeUtil/releases)列出带标签的版本及桌面附件。网站还提供对应代码版本已通过验证的构建。反馈问题时可附下载旁标明的提交版本。
