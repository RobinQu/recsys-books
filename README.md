# RecSys Atlas

RecSys Atlas 是一套从协同过滤到生成式推荐的中文交互式教程。它把召回与排序的技术演进、论文原文证据、33 个可执行 Notebook、章节源码、真实数据实验和工业实践放进同一个 Python Web 应用；论文、数据集与 Notebook 总目录集中收录在首页附录。

3.0 先以总览串起六门共性课程，再进入数据与实验管线；算法页只展开论文新增数学，其余概念精确回链到 3.0 稳定锚点。第三章随后按“章节导读 → 独立算法 → 指标总结”组织，覆盖 CF、MF、FM、GBDT+LR，DSSM、MIND、SASRec，DeepFM、DIN、DIEN，以及 MMoE、PLE。第四章解释生成式召回、生成式排序和召排融合，并提供 OpenOneRec 与 DLRM HSTU 实战。首页 Appendix A.4 只提供有界数学知识图，不重复建立课程目录。

## 1. 初始化论文与数据资源

资源清单位于 `config/resources.json`，实际文件写入已被 Git 忽略的 `resources/`。在项目根目录执行：

```bash
# 下载并校验教程必需的论文 PDF 和 EmbedPDF 阅读器
python scripts/init_resources.py --strict

# 只验证，不访问网络
python scripts/init_resources.py --verify --strict

# 深度章节的完整 Amazon Reviews 2023 数据
python scripts/init_resources.py --include-optional --kind datasets --id amazon-video-games-full

# 排序、多目标与 HSTU 使用的完整 KuaiRand-Pure
python scripts/init_resources.py --include-optional --kind datasets --id kuairand-pure-full
```

Hugging Face gated 数据需先在网页接受许可并设置 `HF_TOKEN`。Google Scholar 只用于发现论文元数据；下载实际来自 arXiv、Hugging Face、作者/机构公开地址或清单中明确的直链。初始化器可重复执行，已就绪文件不会重复下载；应用启动会验证资源状态，Docker build 会先复用本地资源再补齐必需项。

## 2. 启动阅读与实验环境

```bash
docker compose up --build web jupyter ide
```

生成式推荐默认要求 NVIDIA CUDA。GPU 主机使用 Compose 覆盖文件，把基础镜像换成 `Dockerfile.cuda` 构建的 CUDA 版 PyTorch 镜像（本地标签 `recsys-atlas-cuda`），并把设备传入 Web、Jupyter 和测试容器：

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build web jupyter ide
```

没有 CUDA 的主机仍可启动普通服务，但生成式章节的“可交互执行”会被禁用；CPU 自动化测试只验证数据契约、张量形状、前向路径和约束解码，不执行完整精度验收。

- 教程 Web：<http://localhost:8010>
- JupyterLab：<http://localhost:8889/lab?token=recsys>
- 浏览器 IDE：<http://localhost:8090>

Jupyter 服务默认设置 `RECSYS_PROFILE=full`。如果完整数据尚未初始化，相应 Notebook 会给出准确的初始化命令。Web 应用不会在普通启动时静默下载大型数据。

code-server 免登录但仅绑定 `127.0.0.1`，不要直接暴露到公网。每个算法页的 IDE 链接会打开 `chapter_code/<slug>/train.py`；Python 解释器、PyTorch、Torch-RecHub、sklearn、XGBoost、BasedPyright 和 Ruff 均已配置，支持跳转定义、诊断和源码追踪。

## 3. 数据与实验口径

- `smoke`：使用仓库内确定性的真实数据切片，面向 CPU、CI 和代码链路验证；不随机制造行为、曝光或标签。
- `full`：经典章节仍可使用易读的小数据；DSSM/MIND/SASRec 使用完整 Amazon Reviews 2023，DeepFM/DIN/DIEN/MMoE/PLE/HSTU 使用完整 KuaiRand-Pure，生成式章节按官方许可接入 RecIF/Yambda 等资源。
- 生成式 CUDA 验收：默认训练路径使用 CUDA、混合精度和 TF32；完整精度测试仅在 `torch.cuda.is_available()` 为真时执行。

Notebook 的结果会写入 `results/chapter_*/*.json`，章节总结读取同一批产物。论文报告值、项目实测值和生产预期在叙事中严格分开；smoke 指标不是公开榜单成绩。

## 4. 开发教程

依赖与 Python 版本由 [uv](https://docs.astral.sh/uv/) 管理：`pyproject.toml` 声明依赖与 `requires-python`，`uv.lock` 锁定全部平台解析结果，`.python-version` 固定解释器版本。本地开发只需：

```bash
uv sync          # 按 uv.lock 创建/更新 .venv（含 pytest、ruff）
uv run pytest -q # 或 uv run python scripts/build_previews.py
```

Docker 镜像使用同一套锁文件：`uv sync --locked` 把环境安装到 `/opt/venv`（位于 `/workspace` 之外，bind mount 不会遮蔽），Web、Jupyter、测试与浏览器 IDE 共享完全一致的解释器与依赖。

算法专属的结构、数据整理、训练、推理和测试放在 `chapter_code/<slug>/`；`recsys_lab/` 只保留跨章节公共能力。Notebook 由 `scripts/generate_notebooks.py`、`scripts/tutorial_deep_specs.py` 等源文件生成，因此应先修改生成源，再运行：

```bash
python scripts/generate_chapter_code.py
python scripts/generate_notebooks.py
python scripts/generate_model_diagrams.py
python scripts/build_previews.py
```

论文到正文的锚点在 `config/paper_evidence.json` 中维护。每个锚点包含正文关键字、论文 ID、页码、可搜索短语、短引文和边界解释，并在导读正文中派生多个可点击的下划线语义入口。模型结构优先使用 `config/paper_figures.json` 登记的论文原图裁切并保留 Figure/页码出处；论文没有合适结构图时才生成原创 SVG。

详细内容约束、资源策略和页面结构见 `AGENTS.md` 与 `docs/TUTORIAL_REQUIREMENTS.md`。

## 5. 测试与质量门禁

```bash
docker compose run --rm test
```

该命令会生成代码与 Notebook，运行 API/资源/算法效果测试，以 smoke 档执行全部 Notebook，并重新构建 nbconvert 预览。可单独运行：

```bash
docker compose run --rm test python -m ruff check chapter_code recsys_lab app scripts
docker compose run --rm test python scripts/init_resources.py --verify --strict
```

论文效果对照使用独立的 full profile；它会下载并严格校验所需完整资源、验证完整测试集行数，并执行已登记的正式 Notebook。该任务可能下载数 GB 数据并运行较长时间：

```bash
docker compose --profile full run --rm test-full
```

`smoke` 结果不得与论文表格相减；Notebook 只有在 full 数据名称与复现协议匹配时才显示数值 gap，否则明确标记 `NOT COMPARABLE`。

推送 `main` 后，GitHub Actions 会构建 CPU（`Dockerfile`）与 CUDA（`Dockerfile.cuda`）两套 Web 镜像以及 IDE 镜像，并推送到 GitHub Container Registry（`recsys-atlas`、`recsys-atlas-cuda`、`recsys-atlas-ide`）。仓库需要允许 Actions 写入 Packages；工作流使用内置 `GITHUB_TOKEN`，无需提交凭据。

## 6. 贡献检查表

1. 一个算法一个 Notebook，公式中的新符号先解释再使用。
2. 给出论文实验设计、真实训练/推理/测试、baseline 和失败边界。
3. 深度算法默认使用适合其任务的完整新数据，不把 MovieLens 用于所有章节。
4. 新论文先登记资源清单与证据锚点，再链接正文。
5. 运行 Compose 测试并检查明暗主题、窄屏布局、PDF 跳页和 Jupyter/IDE 入口。

---
目前章节页面的notebook预览版本页面中，使用了 iframe 对 notebook 页面进行嵌入。但在iframe里部分链接引用了本教程的其他页面，然后导致了双重页面框架的展示。

比如，[3.0章节](http://localhost:8010/notebooks/3_1_math_foundations) 点击 notebook 里 “一行数据、特征与标签” ，会打开完整版本”3.2“的页面，导致出现了两个侧边栏等内容。

请思考优化方案，比如每个章节的页面能够感知是否处于嵌入模式，并自动隐藏sidebar、topbar 等附属模块。


---

http://localhost:8010/notebooks/3_2_data_ml_basics#implicit-feedback

该notebook的预览结果里有大量 warning 日志，且图表上的中文字符显示未乱码。请修复。

<KERNEL_TEMP>:8: UserWarning: Glyph 30495 (\N{CJK UNIFIED IDEOGRAPH-771F}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 23454 (\N{CJK UNIFIED IDEOGRAPH-5B9E}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 34892 (\N{CJK UNIFIED IDEOGRAPH-884C}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 25968 (\N{CJK UNIFIED IDEOGRAPH-6570}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 19982 (\N{CJK UNIFIED IDEOGRAPH-4E0E}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 20307 (\N{CJK UNIFIED IDEOGRAPH-4F53}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 19981 (\N{CJK UNIFIED IDEOGRAPH-4E0D}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 26159 (\N{CJK UNIFIED IDEOGRAPH-662F}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 21516 (\N{CJK UNIFIED IDEOGRAPH-540C}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 19968 (\N{CJK UNIFIED IDEOGRAPH-4E00}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 20010 (\N{CJK UNIFIED IDEOGRAPH-4E2A}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 27010 (\N{CJK UNIFIED IDEOGRAPH-6982}) missing from font(s) DejaVu Sans.
  plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
<KERNEL_TEMP>:8: UserWarning: Glyph 24565 (\N{CJK UNIFIED IDEOGRAPH-5FF5}) missing from font(s) DejaVu Sans.

