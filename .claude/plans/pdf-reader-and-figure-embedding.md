# 计划：PDF 阅读器自适应 + 论文图嵌入 notebook + 精简导读页

## 背景
- 详情页 `app/templates/notebook_shell.html` 有 4 个 tab：论文导读 / 实验预览 / 可交互执行 / 代码实现。
- 论文导读 与 实验预览 各自右侧嵌入一个 EmbedPDF 阅读器 iframe（`paper_reader.html?embedded=1`）。
- EmbedPDF 用 `ZoomMode.FitPage` 初始化，容器 resize 时不重新适配 → 页面宽度变化后 PDF 不跟随、留白/字小。
- 深度 notebook 的 `Model Structure & Formula Walkthrough` 小节只有公式与 ASCII 流程，缺论文原图。
- `Paper & Context` 小节里有 `**本地原文：** [在站内 PDF 阅读器打开...](/papers/{id})` 跳转链接。

## 决策（已与用户确认）
- 「无论文导读」页面 = 3.0 章节(2) + 3.1–3.4 导读(4) + 4.0 导读(1) = **7 个**；总结页与算法详情页保留论文导读。
- 论文图嵌入 notebook 用 **`/static/paper-figures/{id}.webp` 绝对路径**（Web 预览显示；JupyterLab 显示 alt，可接受）。

---

## 改动 1：EmbedPDF 跟随容器宽度（`app/templates/paper_reader.html`）
- 默认 zoom 由 `ZoomMode.FitPage` 改为 **`ZoomMode.FitWidth`**（已确认导出存在 `FitWidth="fit-width"`）。
- `revealEvidence` 与新增 resize 回调都调用 `setZoomLevel(ZoomMode.FitWidth)`（保留 `zoomTo` 兜底）。
- 把 `scopedZoom`、`ZoomMode` 提到 try 块作用域内可被回调访问；新增：
  - `window.addEventListener('resize', ...)`（debounce ~180ms）
  - `new ResizeObserver(...).observe(target)`（覆盖 tab 切换 display:none→block、栅格断点变化）
- 仅改 reader 脚本，覆盖独立页 `/papers/{id}` 与所有嵌入 iframe（论文导读 tab）。

## 改动 2：实验预览 去 PDF 阅读器 + 去 annotation 跳转（`notebook_shell.html` + `notebook.css`）
- `data-panel="preview"` 面板重写为单栏 `.experiment-preview-pane`：标题 + `math-route` + 全宽 `#notebook-preview-frame`。
- 删除右侧 `paper-guide-viewer` iframe、`figure-evidence-button`、`paper-annotation-card`（这些是跳 PDF 的 annotation 链接）。
- 论文导读 tab **保持不变**（仍保留 PDF 阅读器与下划线 annotation，那是论文阅读页）。
- CSS：新增 `.experiment-preview-pane`（带边框卡片、notebook iframe 高 `calc(100vh - 340px)`）；删除不再使用的 `.experiment-preview-prose` / `experiment-guide-split` 标记；保留 `.paper-guide-split` 栅格规则（论文导读仍用，测试依赖）。

## 改动 3：论文图 + 关键模块嵌入 notebook 正文（生成器 + 在线 patch）
- 新建 `config/model_layers.json`：把 `app/evidence.py` 里的 `MODEL_LAYERS`（按 paper_id）迁到此 JSON。
- `app/evidence.py`：删除内联 `MODEL_LAYERS`，改为 `load_model_layers()` 读 JSON（行为不变，供 web 论文导读卡片与生成器共用）。
- `scripts/tutorial_deep_specs.py`：
  - 模块级新增 `figure_markdown(paper_id)`：读 `paper_figures.json`(label/page) + `model_layers.json`，返回
    ```
    ![{label}](/static/paper-figures/{id}.webp)
    > **论文原图节选** · {label} · PDF p.{page}。下图直接截取自原文…

    ### 关键模块
    - **{name}**：{explanation}
    …
    ```
  - `paper_evidence` 提到模块级（patch 脚本复用）。
  - Model Structure 单元格：在 `## Model Structure & Formula Walkthrough` 与推导之间插入 `figure_markdown(paper_id)`。
  - Paper & Context 单元格：删去 `**本地原文：** [在站内 PDF 阅读器打开并保留证据页码](/papers/{paper_id})` 行（保留 arXiv 来源链接）。
- 新建 `scripts/embed_paper_figures.py`（幂等）：对 10 个深度 notebook 原地 patch——注入 figure 块、移除「本地原文」链接，**保留全部 code 单元格输出**（不重新执行）。
- 运行 `python scripts/embed_paper_figures.py` → `python scripts/build_previews.py`（仅 nbconvert，不执行）。

## 改动 4：7 个页面去论文导读（`app/content.py` + `app/main.py` + `notebook_shell.html`）
- `content.py` 新增 `notebook_has_paper_guide(slug)`：foundations 章节全部 notebook + classic/retrieval/ranking/multitask/generative 的 opening → False；其余 True。
- `main.py` notebook_preview context 增加 `"show_paper_guide": notebook_has_paper_guide(slug)`。
- 模板：`{% set default_mode = 'paper' if show_paper_guide else 'preview' %}`；
  - `show_paper_guide=False` 时不渲染论文导读 tab 按钮 / `data-panel="paper"` 面板；实验预览按钮与面板默认 active。
  - 各 panel 的 `active`/`hidden` 由 `default_mode` 决定。

## 改动 5：测试更新（`tests/test_app.py`）
- `test_every_detail_has_four_modes_and_a_two_column_paper_guide` → 重写为按角色断言：
  - 有论文导读的 19 页：4 tab、`data-panel="paper"`、`id="paper-guide-frame"`、`paper-guide-split`、`paper-annotation`≥8、`data-paper-frame`==1。
  - 无论文导读的 7 页：3 tab、无 `data-panel="paper"`、无 `paper-guide-split`、无 `paper-annotation`、无 `data-paper-frame`。
  - 两类都有 `src="/notebooks/{slug}/content"`、`数学分工`、`/notebooks/3_0_math_foundations`；保留 CSS 断言。
- `test_local_paper_reader_and_evidence_api`：`ZoomMode.FitPage` → `ZoomMode.FitWidth`。
- 其余 test_app / test_notebook_content 断言（章节标题、token）不受影响（已核对）。

## 验证
1. `python scripts/embed_paper_figures.py` && `python scripts/build_previews.py`
2. `ruff check app scripts tests` + `python -m py_compile` 改动文件
3. `pytest tests/test_app.py tests/test_notebook_content.py tests/test_paper_figures.py -q`
4. 抽查：`notebook_previews/3_2_1_dssm.html` 含 `/static/paper-figures/dssm.webp` 与「关键模块」；`/notebooks/3_0_math_foundations` 无「论文导读」tab；`/notebooks/3_2_1_dssm` 实验预览无 PDF iframe。
5. reader 视觉核对（公式/图页 + 窄视口）受限于无浏览器，以代码逻辑 + 测试保证（ResizeObserver + FitWidth）。

## 风险/回滚
- patch 脚本幂等且只改 markdown 单元格，不碰 code 输出；可重复运行。
- 生成器已同步更新，未来 `generate_notebooks.py` + `execute_notebooks.py` 全量再生不会丢图。
- 图用 `/static/` 绝对路径：Web 预览正常；JupyterLab 交互页图片不渲染（alt 文本可见），符合用户选择。
