"""Generate a small, consistent Python package for every tutorial chapter."""
from __future__ import annotations

from pathlib import Path
import sys
import ast
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.content import CHAPTER_CODE_NOTEBOOK_KINDS, NOTEBOOKS

OUTPUT = ROOT / "chapter_code"

MODEL_IMPORTS = {
    "5_2_dssm": ("torch_rechub.models.matching", "DSSM"),
    "5_3_mind": ("torch_rechub.models.matching", "MIND"),
    "5_4_sasrec": ("torch_rechub.models.matching", "SASRec"),
    "6_2_deepfm": ("torch_rechub.models.ranking", "DeepFM"),
    "6_3_din": ("torch_rechub.models.ranking", "DIN"),
    "6_4_dien": ("torch_rechub.models.ranking", "DIEN"),
    "7_2_mmoe": ("torch_rechub.models.multi_task", "MMOE"),
    "7_3_ple": ("torch_rechub.models.multi_task", "PLE"),
    "8_3_dlrm_hstu_practice": ("torch_rechub.models.generative", "HSTUModel"),
}

RUNNERS = {
    "5_2_dssm": "run_dssm", "5_3_mind": "run_mind", "5_4_sasrec": "run_sasrec",
    "6_2_deepfm": "run_deepfm", "6_3_din": "run_din", "6_4_dien": "run_dien",
    "7_2_mmoe": "run_mmoe", "7_3_ple": "run_ple",
    "8_2_openonerec_practice": "run_openonerec", "8_3_dlrm_hstu_practice": "run_hstu",
}

RUN_BLOCKS = {
    "5_2_dssm": ["run_dssm"],
    "5_3_mind": ["_mind_rows", "run_mind"],
    "5_4_sasrec": ["_sequence_windows_from_sequences", "run_sasrec"],
    "6_2_deepfm": ["_ranking_fields", "run_deepfm"],
    "6_3_din": ["_run_sequence_ranker", "run_din"],
    "6_4_dien": ["_run_sequence_ranker", "run_dien"],
    "7_2_mmoe": ["_multitask_view", "_run_multitask", "run_mmoe"],
    "7_3_ple": ["_multitask_view", "_run_multitask", "run_ple"],
    "8_2_openonerec_practice": ["_semantic_catalog", "run_openonerec"],
    "8_3_dlrm_hstu_practice": ["_sequence_windows_from_sequences", "run_hstu"],
}

TRAIN_IMPORTS = {
    "5_2_dssm": "from torch_rechub.basic.features import SparseFeature\nfrom torch_rechub.models.matching import DSSM",
    "5_3_mind": "from torch_rechub.basic.features import SequenceFeature, SparseFeature\nfrom torch_rechub.models.matching import MIND",
    "5_4_sasrec": "from torch_rechub.basic.features import SequenceFeature\nfrom torch_rechub.models.matching import SASRec",
    "6_2_deepfm": "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import log_loss\nfrom torch_rechub.basic.features import SparseFeature\nfrom torch_rechub.models.ranking import DeepFM",
    "6_3_din": "from sklearn.metrics import log_loss\nfrom torch_rechub.basic.features import SequenceFeature, SparseFeature\nfrom torch_rechub.models.ranking import DIN",
    "6_4_dien": "from sklearn.metrics import log_loss\nfrom torch_rechub.basic.features import SequenceFeature, SparseFeature\nfrom torch_rechub.models.ranking import DIEN",
    "7_2_mmoe": "from sklearn.linear_model import LogisticRegression\nfrom torch_rechub.basic.features import DenseFeature\nfrom torch_rechub.models.multi_task import MMOE",
    "7_3_ple": "from sklearn.linear_model import LogisticRegression\nfrom torch_rechub.basic.features import DenseFeature\nfrom torch_rechub.models.multi_task import PLE",
    "8_2_openonerec_practice": "from .model import TinyListGenerator",
    "8_3_dlrm_hstu_practice": "from torch_rechub.models.generative import HSTUModel",
}

MODEL_GUIDES = {
    "5_2_dssm": ["user tower 与 item tower 分别编码，才能离线预计算物品向量", "两个塔输出同维向量，余弦/内积越大表示偏好越强", "训练使用点击标签；线上召回用用户向量查询 ANN 索引"],
    "5_3_mind": ["行为序列先经动态路由聚成多个兴趣向量", "每个兴趣向量独立召回，最后合并候选", "label-aware attention 只用于训练时选择与目标物品最相关的兴趣"],
    "5_4_sasrec": ["因果 mask 保证位置 t 只能读取过去行为", "位置编码保存行为顺序，self-attention 选择相关历史", "最后位置的隐藏状态作为 next-item 召回用户向量"],
    "6_2_deepfm": ["FM 分支显式计算一阶与二阶交叉", "DNN 分支学习更高阶非线性交互", "两条分支共享 embedding，避免同一特征学习两套不一致表示"],
    "6_3_din": ["候选物品作为 query，对历史行为逐条计算相关性", "加权历史表示会随候选变化", "拼接用户、候选与兴趣表示后预测点击概率"],
    "6_4_dien": ["GRU 从行为序列提取逐时刻兴趣状态", "辅助损失要求相邻兴趣能够区分真实下一行为和负样本", "AUGRU 用候选相关性控制兴趣状态更新"],
    "7_2_mmoe": ["多个 expert 学习不同共享模式", "每个任务的 gate 生成独立专家权重", "任务 tower 只读取自己的混合结果并输出对应目标"],
    "7_3_ple": ["共享 expert 与任务专属 expert 明确分开", "每层 gate 控制信息流向，减少不相关任务的负迁移", "多层渐进抽取后再进入各任务 tower"],
    "8_2_openonerec_practice": ["item 先被编码为多级 Semantic ID", "生成器逐 token 输出推荐列表", "解码时必须用合法目录约束，避免生成不存在的物品"],
    "8_3_dlrm_hstu_practice": ["item embedding 将高基数 ID 映射为稠密向量", "HSTU block 针对非平稳长行为流建模", "每个位置预测下一物品，最后位置用于在线推荐"],
}

CLASSIC_MODELS = {
    "4_2_collaborative_filtering": '''import numpy as np

class ItemCF:
    """Item-based collaborative filtering using cosine similarity."""
    def fit(self, interactions):
        values = np.asarray(interactions, dtype=np.float64)
        # 每一列是一件物品；列向量记录哪些用户与它交互过。
        norms = np.linalg.norm(values, axis=0, keepdims=True)
        # 矩阵乘法一次得到所有物品对的共同用户数，再除以长度得到余弦相似度。
        denominator = np.maximum(norms.T @ norms, 1e-12)
        self.similarity = np.nan_to_num((values.T @ values) / denominator)
        # 不允许物品把自己当作邻居，否则自身相似度 1 会淹没其他候选。
        np.fill_diagonal(self.similarity, 0.0)
        return self

    def score(self, interactions):
        # 用户历史 × 物品相似度表，得到该用户对所有候选物品的分数。
        return np.asarray(interactions, dtype=np.float64) @ self.similarity
''',
    "4_3_matrix_factorization": '''import torch

class BiasMF(torch.nn.Module):
    """Global mean + user/item bias + latent-vector inner product."""
    def __init__(self, users, items, factors=16):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(users, factors)
        self.item_embedding = torch.nn.Embedding(items, factors)
        self.user_bias = torch.nn.Embedding(users, 1)
        self.item_bias = torch.nn.Embedding(items, 1)
        # 全局均值描述总体评分尺度；两个 bias 修正用户/物品的系统性偏差。
        self.global_mean = torch.nn.Parameter(torch.zeros(()))

    def forward(self, user_id, item_id):
        # 对应位置逐元素相乘再求和，就是用户向量与物品向量的内积。
        interaction = (self.user_embedding(user_id) * self.item_embedding(item_id)).sum(-1)
        return self.global_mean + self.user_bias(user_id).squeeze(-1) + self.item_bias(item_id).squeeze(-1) + interaction
''',
    "4_4_factorization_machine": '''import torch

class FactorizationMachine(torch.nn.Module):
    """Linear term plus all second-order feature interactions in O(kd)."""
    def __init__(self, fields, factors=16):
        super().__init__()
        self.linear = torch.nn.Embedding(fields, 1)
        self.embedding = torch.nn.Embedding(fields, factors)

    def forward(self, feature_ids):
        # vectors 的形状为 [batch, field, factor]，每个稀疏特征都有一个隐向量。
        vectors = self.embedding(feature_ids)
        linear = self.linear(feature_ids).sum(1).squeeze(-1)
        # 恒等式将两两内积从 O(field²) 化为 O(field)，避免显式双重循环。
        pairwise = 0.5 * ((vectors.sum(1) ** 2 - (vectors ** 2).sum(1)).sum(1))
        return linear + pairwise
''',
    "4_5_gbdt_lr": '''from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def build_models(seed=2026):
    """The tree creates leaf features; LR calibrates their joint score."""
    # GBDT 负责非线性分桶/组合；训练后应把叶节点编号 one-hot 化再输入 LR。
    return GradientBoostingClassifier(random_state=seed), LogisticRegression(max_iter=300)
''',
    "4_6_word2vec": '''import torch

class SkipGram(torch.nn.Module):
    """Skip-gram with negative sampling: a center item predicts its context items."""
    def __init__(self, num_items, dim=32):
        super().__init__()
        # 中心嵌入与上下文嵌入分离，是 word2vec 的常规做法；负采样时只更新相关行。
        self.center = torch.nn.Embedding(num_items, dim)
        self.context = torch.nn.Embedding(num_items, dim)
        torch.nn.init.uniform_(self.center.weight, -0.01, 0.01)
        torch.nn.init.zeros_(self.context.weight)

    def forward(self, center_ids, context_ids):
        # 中心向量与上下文向量的内积越大，两物品越可能在同一序列共现。
        return (self.center(center_ids) * self.context(context_ids)).sum(-1)


def negative_sampling_loss(score_pos, score_neg):
    """BCE on logits: pull positives together, push negatives away."""
    # 正样本希望 σ(score)->1，负样本希望 σ(score)->0；等价于负采样目标函数。
    pos = torch.nn.functional.binary_cross_entropy_with_logits(score_pos, torch.ones_like(score_pos))
    neg = torch.nn.functional.binary_cross_entropy_with_logits(score_neg, torch.zeros_like(score_neg))
    return pos + neg
''',
}


def model_source(slug: str, title: str) -> str:
    if slug in CLASSIC_MODELS:
        return f'"""Model structure for {title}."""\n\n{CLASSIC_MODELS[slug]}'
    if slug in MODEL_IMPORTS:
        module, name = MODEL_IMPORTS[slug]
        guide = "\n".join(f"# - {line}" for line in MODEL_GUIDES[slug])
        return f'''"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from {module} import {name}

# 阅读结构定义时先抓住以下三点：
{guide}
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = {name}

def model_class():
    return MODEL_CLASS
'''
    if slug == "8_2_openonerec_practice":
        return '''"""Small list generator used to explain OpenOneRec's list-level objective."""
import torch

class TinyListGenerator(torch.nn.Module):
    """Autoregressive proxy: token embedding → GRU state → next-token logits."""
    def __init__(self, vocabulary_size: int, hidden: int = 32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden)
        self.gru = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, vocabulary_size)

    def forward(self, tokens):
        states, _ = self.gru(self.embedding(tokens))
        return self.head(states)

# 代理模型保留“逐 token 生成列表”的关键接口；完整 OpenOneRec 还包含奖励模型与偏好对齐。
# 输入是 Semantic ID 前缀，输出是下一层 code token 的 logits。
# 推理时必须用合法 ID trie 屏蔽目录中不存在的 token。
MODEL_CLASS = TinyListGenerator
'''
    return f'''"""Executable concepts for {title}.

This chapter is a foundation or comparison chapter. Its runnable calculations
live in the Notebook; shared data and experiment modules are shown in the code browser.
"""
MODEL_CLASS = None
'''


def train_source(slug: str) -> str:
    runner = RUNNERS.get(slug)
    if runner:
        source_path = ROOT / "recsys_lab" / "industrial_experiments.py"
        source = source_path.read_text(encoding="utf-8")
        lines = source.splitlines()
        wanted = set(RUN_BLOCKS[slug])
        found = set()
        blocks = []
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
                found.add(node.name)
                blocks.append("\n".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)]))
        implementation = "\n\n".join(blocks)
        missing = wanted - found
        # chapter_code 是实现的可读源头；industrial_experiments.py 已改为 load_runner
        # 委托存根。提取不到真实实现时必须失败，否则会用存根覆盖出不可运行的章节代码。
        if missing or "load_runner" in implementation:
            raise RuntimeError(
                f"{slug}: 无法脚手架 train.py —— {source_path} 中没有 "
                f"{sorted(missing) or sorted(wanted)} 的真实实现（该模块现在只保留委托存根）。"
                f"请从 git 恢复 chapter_code/{slug}/train.py，或先在 {source_path} 中补回实现。"
            )
        implementation = annotate_training(implementation)
        framework_imports = TRAIN_IMPORTS[slug]
        return f'''"""Training and evaluation orchestration for {slug}.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
{framework_imports}
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    real_amazon as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
)

{implementation}

def train_and_evaluate(epochs: int = 4) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return {runner}(epochs=epochs)
'''
    if slug.startswith("4_"):
        return '''"""Classic-model experiment entry on MovieLens."""
from recsys_lab.experiments import run_classic

def train_and_evaluate(epochs: int = 8) -> dict:
    # 统一入口会读取 MovieLens、按时间切分，并返回各经典基线的可比指标。
    return run_classic(epochs=epochs)
'''
    return '''"""Narrative chapter: execute its Notebook to run the demonstrations."""

def train_and_evaluate(epochs: int = 1) -> dict:
    return {"kind": "foundation_or_summary", "epochs": epochs}
'''


def annotate_training(source: str) -> str:
    """Insert compact learning landmarks without changing executable behavior."""
    landmarks = [
        ("seed_everything();", "# 1) 固定参数初始化，并读取本章指定的真实数据切片。"),
        ("model =", "# 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。"),
        ("optimizer =", "# 3) optimizer 只在训练阶段更新参数；推理阶段不应再调用它。"),
        ("losses = _train_binary(", "# 4) 公共训练循环执行 forward、二元交叉熵、backward 和 optimizer.step。"),
        ("for _ in range(epochs):", "# 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。"),
        ("with torch.no_grad():", "# 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。"),
        ("return {", "# 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。"),
    ]
    output, used = [], set()
    for line in source.splitlines():
        for marker, comment in landmarks:
            indentation = len(line) - len(line.lstrip())
            if marker == "return {" and indentation != 4:
                continue
            if marker in line and marker not in used:
                output.append(f"{line[:len(line) - len(line.lstrip())]}{comment}")
                used.add(marker)
                break
        output.append(line)
    return "\n".join(output)


def write_chapter(slug: str, title: str) -> None:
    directory = OUTPUT / slug
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "__init__.py": f'"""Python companion package for {title}."""\n',
        "model.py": model_source(slug, title),
        "inference.py": '''"""Reusable inference path, deliberately separated from optimization."""
from __future__ import annotations
import torch

def predict(model, batch):
    """Disable dropout/gradients and return model output for a prepared batch."""
    # eval() 会关闭 dropout，并让 batch normalization 使用已学习的统计量。
    model.eval()
    # no_grad() 表明这里只产生预测，不保存反向传播所需的计算图。
    with torch.no_grad():
        output = model(batch)
    # 移到 CPU 后再交给指标代码或在线服务序列化，避免设备不一致。
    return tuple(value.detach().cpu() for value in output) if isinstance(output, tuple) else output.detach().cpu()
''',
        "test_model.py": '''"""Chapter-local smoke test; repository tests enforce stronger effect thresholds."""
from train import train_and_evaluate

def test_training_pipeline_returns_observed_results():
    # 只跑 1 个 epoch 验证数据、模型、loss、推理和指标整条链路能够连通。
    result = train_and_evaluate(epochs=1)
    assert isinstance(result, dict) and result
    assert result.get("randomly_fabricated_rows", 0) == 0
''',
    }
    if slug in {"8_2_openonerec_practice", "8_3_dlrm_hstu_practice"}:
        files["test_model.py"] = '''"""CUDA-first chapter test with an explicit CPU-only basic fallback."""
import torch
from train import train_and_evaluate

def test_training_pipeline_returns_observed_results():
    result = train_and_evaluate(epochs=1, cpu_smoke=not torch.cuda.is_available())
    assert isinstance(result, dict) and result
    assert result.get("randomly_fabricated_rows", result.get("dataset", {}).get("randomly_fabricated_rows", 0)) == 0
'''
    # 已有 train.py 是读者可见的实现源头，只有缺失时才脚手架；
    # 提取不到真实实现时 train_source 会响亮失败，而不是产出委托存根。
    if not (directory / "train.py").exists():
        files["train.py"] = train_source(slug)
    for name, source in files.items():
        path = directory / name
        # Chapter Python files are maintained as the readable source of truth.
        # The generator scaffolds a new chapter but never overwrites reader-facing
        # implementations with a hidden central template.
        if not path.exists():
            path.write_text(source.rstrip() + "\n", encoding="utf-8")
        elif name == "test_model.py":
            migrated = path.read_text(encoding="utf-8").replace(
                "from train import train_and_evaluate",
                "from .train import train_and_evaluate",
            )
            path.write_text(migrated, encoding="utf-8")
    vscode = directory / ".vscode"
    vscode.mkdir(exist_ok=True)
    settings = {
        # Resolve the repository and framework sources copied into the IDE image.
        "basedpyright.analysis.extraPaths": ["../..", "/opt/venv/lib/python3.11/site-packages"],
        "basedpyright.analysis.useLibraryCodeForTypes": True,
        "basedpyright.analysis.diagnosticMode": "openFilesOnly",
        # RecHub currently ships incomplete type metadata. Keep navigation and
        # imports, while Ruff/compile gates own correctness diagnostics.
        "basedpyright.analysis.typeCheckingMode": "off",
        "python.defaultInterpreterPath": "/opt/venv/bin/python",
        "python.analysis.extraPaths": ["../..", "/opt/venv/lib/python3.11/site-packages"],
        "python.terminal.activateEnvironment": False,
        # Ruff and BasedPyright share one interpreter and one repository-level rule set.
        "ruff.configuration": "/home/coder/project/pyproject.toml",
        "ruff.lint.enable": True,
        "ruff.nativeServer": "on",
        "workbench.startupEditor": "none",
        "editor.definitionLinkOpensInPeek": False,
        # Keep the editor canvas uncluttered. Users can reopen it from View → Appearance.
        "workbench.secondarySideBar.defaultVisibility": "hidden",
        # Follow browser/OS color preference; VS Code's theme picker remains available.
        "window.autoDetectColorScheme": True,
        "workbench.preferredDarkColorTheme": "Default Dark Modern",
        "workbench.preferredLightColorTheme": "Default Light Modern",
    }
    (vscode / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pyright = {
        "include": ["."],
        "extraPaths": ["../..", "/opt/venv/lib/python3.11/site-packages"],
        "useLibraryCodeForTypes": True,
        "typeCheckingMode": "off",
    }
    (directory / "pyrightconfig.json").write_text(json.dumps(pyright, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_chapter_code_notebooks(notebooks: list[dict]) -> list[dict]:
    """Return notebook records that own an algorithm-style companion package."""
    return [notebook for notebook in notebooks if notebook.get("kind") in CHAPTER_CODE_NOTEBOOK_KINDS]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    package = OUTPUT / "__init__.py"
    if not package.exists():
        package.write_text('"""Executable, chapter-local recommendation examples."""\n', encoding="utf-8")
    eligible_notebooks = select_chapter_code_notebooks(NOTEBOOKS)
    for notebook in eligible_notebooks:
        write_chapter(notebook["slug"], notebook["title"])
    print(f"generated {len(eligible_notebooks)} chapter packages in {OUTPUT}")


if __name__ == "__main__":
    main()
