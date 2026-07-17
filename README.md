# RecSys Atlas

交互式推荐算法技术图谱，包含 26 个可执行 Notebook、nbconvert 预览、JupyterLab、code-server 源码 IDE、框架/数据集调研和 Docker Compose 自动化测试。算法页面以 Tab 直接切换阅读预览、可交互执行和实现源码；SASRec 已归入 3.2 召回章节。

## 启动

```bash
docker compose up --build web jupyter ide
```

- Web 应用：<http://localhost:8010>
- JupyterLab：<http://localhost:8889/lab?token=recsys>
- 浏览器 IDE：<http://localhost:8090>（免登录，仅绑定 `127.0.0.1`；不要改为公网监听）

每个算法页的“独立 IDE / 在 IDE 中编辑”入口会直接打开该章节的独立源码目录。首次打开和已经存在 IDE 会话时均可使用。

IDE 内置 Python 扩展与 BasedPyright，并安装和 Web/Jupyter 完全一致的 Python 3.11、PyTorch、Torch-RecHub、sklearn、XGBoost 等依赖。解释器固定为 `/usr/local/bin/python`；将光标放到 `DSSM` 等符号上，使用 `F12` 或 `Cmd/Ctrl + 单击` 可跳到定义。进入章节时默认打开该目录的 `train.py`，Secondary Sidebar 默认隐藏。教程页面会优先跟随浏览器/系统的明暗偏好，也可用顶栏按钮手动切换并记住选择；IDE 会跟随系统配色，也可使用其主题选择器覆盖。

## 章节源码

每个 Notebook 都有对应的 `chapter_code/<slug>/` Python 目录，包含模型结构、算法专属的数据整理、训练、推理和章节 smoke test。`recsys_lab/` 只保留跨章节复用的数据加载、通用训练循环、指标和兼容路由，不再隐藏 MIND 序列构造、DIN/DIEN 训练、MMoE/PLE 多任务张量或 Semantic ID 等章节算法实现。页面“实现源码”Tab 还会展示这些公共基础文件、仓库效果测试，以及实际使用的 Torch-RecHub 模型和 Trainer 源文件。

Jupyter 镜像已安装 `ipywidgets`、`widgetsnbextension` 与 `jupyterlab_widgets`，因此 `tqdm.auto` 可以正常选择 Notebook 进度条，不会再出现 `IProgress not found`。源码预览保留缩进并自动换行，长注释和长表达式无需横向滚动才能读完。

## 测试

```bash
docker compose run --rm test
```

测试会依次生成 Notebook、运行单元/API/效果阈值测试、执行全部 26 个 Notebook，并重新生成预览。smoke 数据固定且无需网络下载。3.1—4.3 的算法实验会分别写出 `results/chapter_*/*.json`，各章总结再读取这些产物形成横向指标表。

算法 Python 源码同时受两道质量门禁保护：内存编译检查负责捕获语法错误，Ruff 负责捕获非法结构与未定义名称。IDE 使用同一份 `pyproject.toml`，因此编辑器 Problems 面板和 Docker 测试保持一致。可单独执行：

```bash
docker compose run --rm test python -m ruff check chapter_code recsys_lab
```

## 数据运行档

- `RECSYS_PROFILE=smoke`：按任务读取仓库内 MovieLens、Amazon Reviews 2023 Video Games 或 KuaiRand-Pure 的确定性真实数据切片，适合 CPU 与 CI；不制造交互、曝光、标签或序列。
- `RECSYS_PROFILE=full`：接入以上数据的官方完整版本；OpenOneRec 的 RecIF-Bench 当前需要在 Hugging Face 接受许可并认证，Meta DLRM-v3 还需按官方文档准备 GPU/权重。

Notebook 是教程而不是基准榜单；smoke 输出只证明代码路径、损失与指标有效，不能代表公开数据或生产效果。
