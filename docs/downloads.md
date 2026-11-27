# 安装与下载

## 网页版查看器

点击导航栏的「打开查看器」，拖入 trace 或选择整个采集文件夹。文件在浏览器本地处理。也可以直接打开教程中的真实样例。

## 桌面客户端

按操作系统和架构下载下方的构建产物，完整解压。Windows 打开 **ServeUtil.exe**，macOS 打开 **ServeUtil.app**，Linux 运行解压目录内的 **ServeUtil**。桌面窗口可离线读取 JSON、gzip、ServeUtil 报告和原生 Perfetto trace。

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

## 版本记录

[GitHub Releases](https://github.com/HaoyangPing0324/ServeUtil/releases)列出带标签的版本及桌面附件。网站还提供对应代码版本已通过验证的构建。反馈问题时可附下载旁标明的提交版本。
