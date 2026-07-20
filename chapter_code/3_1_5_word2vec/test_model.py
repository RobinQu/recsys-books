"""Chapter-local smoke test; repository tests enforce stronger effect thresholds."""
from .train import train_and_evaluate


def test_training_pipeline_returns_observed_results():
    # 只跑 1 个 epoch 验证数据、模型、loss、推理和指标整条链路能够连通。
    result = train_and_evaluate(epochs=1)
    assert isinstance(result, dict) and result
    assert result.get("randomly_fabricated_rows", 0) == 0
    assert 0 <= result.get("recall@5", 0.0) <= 1
