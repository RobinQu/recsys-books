# RecSys Atlas

交互式推荐算法技术图谱，包含 26 个可执行 Notebook、nbconvert 预览、JupyterLab、code-server 源码 IDE、框架/数据集调研和 Docker Compose 自动化测试。算法页面以 Tab 直接切换阅读预览、可交互执行和实现源码；SASRec 已归入 3.2 召回章节。

## 启动

```bash
docker compose up --build web jupyter ide
```

- Web 应用：<http://localhost:8010>
- JupyterLab：<http://localhost:8889/lab?token=recsys>
- 浏览器 IDE：<http://localhost:8090>（免登录，仅绑定 `127.0.0.1`；不要改为公网监听）

每个算法页的“独立 IDE / 在 IDE 中编辑”入口会先加载编辑器，再自动定位到该章节对应的实现文件。首次打开和已经存在 IDE 会话时均可使用。

## 测试

```bash
docker compose run --rm test
```

测试会依次生成 Notebook、运行单元/API/效果阈值测试、执行全部 26 个 Notebook，并重新生成预览。smoke 数据固定且无需网络下载。3.1—4.3 的算法实验会分别写出 `results/chapter_*/*.json`，各章总结再读取这些产物形成横向指标表。

## 数据运行档

- `RECSYS_PROFILE=smoke`：按任务读取仓库内 MovieLens、Amazon Reviews 2023 Video Games 或 KuaiRand-Pure 的确定性真实数据切片，适合 CPU 与 CI；不制造交互、曝光、标签或序列。
- `RECSYS_PROFILE=full`：接入以上数据的官方完整版本；OpenOneRec 的 RecIF-Bench 当前需要在 Hugging Face 接受许可并认证，Meta DLRM-v3 还需按官方文档准备 GPU/权重。

Notebook 是教程而不是基准榜单；smoke 输出只证明代码路径、损失与指标有效，不能代表公开数据或生产效果。
