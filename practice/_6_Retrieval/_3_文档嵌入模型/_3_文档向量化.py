"""

"""
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import os,dotenv

dotenv.load_dotenv()

embedding_config = {
	"model_name": os.getenv("BGE_MODEL_PATH"),
	"model_kwargs": {
		"device": "mps"  # 强制使用CPU时设为 "cpu"。有GPU的可设为 "cuda" 达到加速，是NVIDIA 显卡专属的计算框架。Apple 为 Silicon 芯片的GPU可设为 "mps"
	},
	"encode_kwargs": {
		"normalize_embeddings": True,  # 规范化向量（提升检索效果）
		"batch_size": 16  # 批量编码，提升效率
	}
}

embeddings = HuggingFaceEmbeddings(**embedding_config)

csv_loader = CSVLoader("../../resources/asset/load/03-load.csv")

docs = csv_loader.load_and_split() # 加载后直接拆分，默认使用RecursiveCharacterTextSplitter

# 文档向量化
vectors = embeddings.embed_documents([d.page_content for d in docs])

print(len(vectors))

#每个chunk的embedding纬度
print(len(vectors[0]))
print(vectors[0][:10])
