"""
	1. 向量数据库本身已经实现了检索功能，如 similarity_search. 该函数通过计算原始查询向量与数据库中存储向量之间的相似度来实现召回。

	2. LangChain 提供更复杂的召回策略，被集成在 Retrievers （检索器）组件中。Retrievers 是一种从大量文档中检索与给定查询相关的文档或信息片段的工具。
		检索器「不需要存储文档」，只需要「检索文档」即可。存储仍在向量数据库中

		Retrievers 执行步骤：
				将输入的查询转换为向量表示；
				在向量存储中 搜索与查询向量最相似的文档向量（通常使用余弦相似度或欧几里得距离）
				返回与查询最相关的文档或文本片段，及它们的相似度得分

"""
import dotenv,os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

# 基础使用 :  Retriever 一般和 VectorStore 配套实现，通过 as_retriever() 方法获取。 as_retriever()  - 基于向量数据库获取 retriever

loader = TextLoader("../../resources/asset/load/09-ai1.txt")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
doc_chunks = splitter.split_documents(docs)

dotenv.load_dotenv()
embedding_config = {
	"model_name":  os.getenv("BGE_MODEL_PATH"),
	"model_kwargs": {
		"device": "mps"  # 强制使用CPU时设为 "cpu"。有GPU的可设为 "cuda" 达到加速，是NVIDIA 显卡专属的计算框架。Apple 为 Silicon 芯片的GPU可设为 "mps"
	},
	"encode_kwargs": {
		"normalize_embeddings": True,  # 规范化向量（提升检索效果）
		"batch_size": 16  # 批量编码，提升效率
	}
}
embeddings_model = HuggingFaceEmbeddings(**embedding_config)

# 获取向量数据库
db = Chroma.from_documents(doc_chunks, embeddings_model)

# 基于向量数据库获取检索器
retriever = db.as_retriever()

res = retriever.invoke("")
print(len(res))
