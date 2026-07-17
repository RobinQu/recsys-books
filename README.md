# RecSys Atlas

交互式推荐算法技术图谱，包含 27 个可执行 Notebook、nbconvert 预览、JupyterLab、框架/数据集调研和 Docker Compose 自动化测试。首页直接链接“大章数学导读 → 独立算法章节 → 实验指标总结”，不再增加章节详情中转层。

## 启动

```bash
docker compose up --build web jupyter
```

- Web 应用：<http://localhost:8010>
- JupyterLab：<http://localhost:8889/lab?token=recsys>

## 测试

```bash
docker compose run --rm test
```

测试会依次生成 Notebook、运行单元/API/效果阈值测试、执行全部 27 个 Notebook，并重新生成预览。smoke 数据固定且无需网络下载。3.1—4.3 的算法实验会分别写出 `results/chapter_*/*.json`，各章总结再读取这些产物形成横向指标表。

## 数据运行档

- `RECSYS_PROFILE=smoke`：仓库内固定版本的 GroupLens MovieLens latest-small 真实数据，适合 CPU、CI 和代码功能验证；不制造交互、标签或序列。
- `RECSYS_PROFILE=full`：接入 MovieLens、Amazon Reviews 2023、Ali-CCP、RecIF-Bench 等原始数据；OpenOneRec 与 Meta DLRM-v3 需要按官方文档准备 GPU/权重与数据许可。

Notebook 是教程而不是基准榜单；smoke 输出只证明代码路径、损失与指标有效，不能代表公开数据或生产效果。
