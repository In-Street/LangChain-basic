"""
		有API 接口的话，调用在线embedding model ，如 OpenAI -> text-embedding-ada-002 、阿里云通义 -> text-embedding-v1
		使用本地开源嵌入模型：BGE  、M3E（轻量化选择）、Qwen-Embedding（高精度中文嵌入）

		pip3 install torch==2.2.0 transformers==4.40.0 sentence-transformers==3.0.1 numpy==1.26.4

		M3E ：
			轻量版 moka-ai/m3e-small 、 进阶版 moka-ai/m3e-base
			通过Hugging Face transformers/sentence-transformers 加载，支持CPU/GPU运行（CPU可跑轻量化模型）

			sentence-transformers 是对 transformers 的高阶封装，内置大量开源嵌入模型（BERT、M3E）。
			可使用 SentenceTransformer 加载模型： model = SentenceTransformer("moka-ai/m3e-small")

			可使用LangChian 提供的 HuggingFaceEmbeddings 对接 sentence-transformers 模型的核心类，将M3E模型封装为LangChain的Embedding组件

		能否使用Apple Silicon GPU的加速框架mps，与M3E/BGE模型本身无直接关系，取决于模型是否基于PyTorch构建且PyTorch版本（>=2.0）适配M1的mps，而非TensorFlow等其他框架。

"""
from huggingface_hub import snapshot_download
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # huggingface 镜像地址

# 首次使用完模型后，在本地会有缓存。后续可直接使用缓存模型，利用 huggingface-hub 的辅助函数获取缓存路径
cache_m3e_model_path = snapshot_download("moka-ai/m3e-small", local_files_only=True)
bge_model_path = "/Users/chengyufei/Downloads/project/self/py/embedding-model/bge-small-zh-v1.5"

embedding_model_config = {
	# "model_name": "moka-ai/m3e-small",  # 使m3e用轻量版模型
	"model_name": cache_m3e_model_path,  # m3e 本地缓存

	# "model_name": "BAAI/bge-small-zh-v1.5",  # 使用bge轻量版模型
	# "model_name": bge_model_path,  # bge 本地缓存

	"model_kwargs": {
		"device": "mps" # 强制使用CPU时设为 "cpu"。有GPU的可设为 "cuda" 达到加速，是NVIDIA 显卡专属的计算框架。Apple 为 Silicon 芯片的GPU可设为 "mps"
	},
	"encode_kwargs": {
		"normalize_embeddings": True,  # 规范化向量（提升检索效果）
		"batch_size": 16 # 批量编码，提升效率
	}
}

embeddings = HuggingFaceEmbeddings(**embedding_model_config)

text = "M3E嵌入模型"
embedding_vectors = embeddings.embed_query(text)  # 句子向量化，query场景

print(f'{len(embedding_vectors)}')
print(f'向量前10位： {embedding_vectors[:10]}')
