"""
		开源向量数据库： chroma、FAISS
		在向量数据库中检索，结果并不是精确的，而是查询和目标向量最为相似的一些向量，具有模糊性

		pip3 install chromadb==0.5.11
"""
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
import os,dotenv

#  1.  TextLoader 加载指定文档
loader = TextLoader(
	file_path="../../resources/asset/load/09-ai1.txt",
	encoding="utf-8",
)
docs = loader.load()

# 2.  文档拆分器，拆分文档
splitter = CharacterTextSplitter(
	chunk_size=1000,
	chunk_overlap=100
)
split_documents = splitter.split_documents(docs)

# 3. 创建嵌入模型。 此处不直接将文档向量化，作为参数传入向量数据库中
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

# 4.  将文档及嵌入模型传入Chroma中，进行数据存储
db = Chroma.from_documents(
	documents=split_documents,
	embedding=embeddings_model,
	persist_directory="../../resources/asset/chroma/1",  # 显示指定存储位置。 不设置此参数时，会存储在内存中
)

# 5. 简单查询
search_res = db.similarity_search("人工智能的概念最早出现在什么时间？")
print(f'搜索总文档数：{len(search_res)}')
print(search_res[0])



# 从本地已存储的数据中进行检索
def search_from_chroma(query:str):
	db_ = Chroma(
		embedding_function=embeddings_model,
		persist_directory="../../resources/asset/chroma/1",
		collection_metadata={"hnsw:space": "cosine"}  # 指定余弦相似度
	)
	count = db_._collection.count()
	print(f'成功加载已有的Chroma数据库，当前文档数量：{count}')

	search_result = db_.similarity_search_with_score(query=query, k=2)
	print(f'现有数据搜索结果：{len(search_result)}')

	for i, (doc, score) in enumerate(search_result):
		print(f"\n【第{i + 1}条】相似度分数：{score:.4f}")
		print(f"文档内容：{doc.page_content}")
		print("-" * 40)


if __name__ == '__main__':
	search_from_chroma("北京有多少地铁线路？")