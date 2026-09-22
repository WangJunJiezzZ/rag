"""
向量化层 —— 可切换 provider, 默认离线。

为什么是 ONNX 而不是 sentence-transformers
------------------------------------------
本项目的硬约束是"整个文件夹搬到另一台 Windows 上要能跑"。
  sentence-transformers  需要 torch(数百 MB~2GB), 且首次运行要联网下模型
  ONNX Runtime           约 40MB, 模型量化后 24MB 可随仓库分发, 首次运行即离线

代价是放弃了 torch 生态的灵活性(微调、批量 GPU 推理), 但本项目不需要。
**依赖体积是部署约束的一部分, 不是次要因素。**

provider
--------
  onnx    models/bge-small-zh/ 下的量化模型, 离线, 推荐
  hash    确定性伪向量, 零依赖。**不是能用的检索器**, 只用于在没装
          onnxruntime 的机器上跑通代码路径与单元测试
  none    显式关闭 dense —— 消融实验的基线行
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .compat import project_root

DEFAULT_MODEL_DIR = "models/bge-small-zh"

# bge 系列的官方建议: 查询侧加指令前缀, 文档侧不加。
# 这一条不加, 检索质量会明显下降 —— 属于"模型用法"而非"调参",
# 但很多实现会漏掉。是否加前缀在本项目里是一个可消融项。
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class BaseEmbedder:
    name = "base"
    dim = 0

    def encode(self, texts: list[str], is_query: bool = False):
        raise NotImplementedError


class ONNXEmbedder(BaseEmbedder):
    """bge-small-zh-v1.5 量化版, 纯 CPU 推理。"""
    name = "onnx"

    def __init__(self, model_dir: str | Path | None = None,
                 max_len: int = 512, use_prefix: bool = True):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir or project_root() / DEFAULT_MODEL_DIR)
        onnx_path = d / "model_quantized.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"未找到模型 {onnx_path}\n"
                f"运行 `python run.py fetch-model` 下载(约 24MB), "
                f"或设置 GRAPHRAG_EMBED=none 退化为纯 BM25。")

        self._np = np
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = min(4, os.cpu_count() or 1)
        self.sess = ort.InferenceSession(str(onnx_path), so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.use_prefix = use_prefix
        self.dim = self.sess.get_outputs()[0].shape[-1] or 512

    def encode(self, texts: list[str], is_query: bool = False, batch: int = 32):
        np = self._np
        if is_query and self.use_prefix:
            texts = [BGE_QUERY_PREFIX + t for t in texts]

        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]        # (B, L, H)
            # bge 用 [CLS] 池化(取第 0 个 token), 不是 mean pooling。
            # 用错池化方式会静默地降低质量 —— 不会报错, 只是分数变差。
            vec = hidden[:, 0, :]
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
            out.append(vec.astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype="float32")


class HashEmbedder(BaseEmbedder):
    """确定性伪向量。**没有语义**, 仅用于在缺依赖的机器上跑通代码路径。"""
    name = "hash"

    def __init__(self, dim: int = 256):
        import numpy as np
        self._np = np
        self.dim = dim

    def encode(self, texts: list[str], is_query: bool = False):
        np = self._np
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for j in range(0, max(len(t) - 1, 1)):
                h = int(hashlib.md5(t[j:j + 2].encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        return out


def get_embedder(kind: str | None = None, **kw) -> BaseEmbedder | None:
    kind = (kind or os.environ.get("GRAPHRAG_EMBED") or "onnx").lower()
    if kind == "none":
        return None
    if kind == "hash":
        return HashEmbedder(**kw)
    try:
        return ONNXEmbedder(**kw)
    except (ImportError, FileNotFoundError) as e:
        print(f"[warn] dense 向量不可用 ({type(e).__name__}); "
              f"退化为纯 BM25。原因: {str(e).splitlines()[0]}")
        return None
