"""
	Text Embedding Models  文档嵌入模型
			将文本编码为向量。 文档写入 和 用户查询 前都会先执行文档嵌入编码，即向量化。

			LangChain 针对向量化模型提供两种接口：
				文档向量化 embed_documents ： 专为「批量文档」设计，支持批量编码提升效率。如知识库文本
				句子向量化embed_query： 专为「查询文本」设计（如用户提问）
"""
import torch,os
from huggingface_hub import snapshot_download

# 检查 MPS 支持。 Apple 为 Silicon 芯片 的GPU加速框架
print(f"PyTorch 版本：{torch.__version__}")
print(f"MPS 是否可用：{torch.backends.mps.is_available()}")
print(f"MPS 是否内置：{torch.backends.mps.is_built()}")


CACHE_MODEL_PATH = snapshot_download("moka-ai/m3e-small", local_files_only=True)
print(f"自动获取的缓存路径：{CACHE_MODEL_PATH}")

print(os.getenv("BGE_MODEL_PATH"))