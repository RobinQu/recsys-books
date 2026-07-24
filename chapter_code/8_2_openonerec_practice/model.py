"""Small list generator used to explain OpenOneRec's list-level objective."""
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
