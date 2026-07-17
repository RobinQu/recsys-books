"""Classic-model experiment entry on MovieLens."""
from recsys_lab.experiments import run_classic

def train_and_evaluate(epochs: int = 8) -> dict:
    # 统一入口会读取 MovieLens、按时间切分，并返回各经典基线的可比指标。
    return run_classic(epochs=epochs)
