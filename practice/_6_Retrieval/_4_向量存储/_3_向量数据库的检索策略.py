import dotenv,os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
db = Chroma(
	embedding_function = embeddings_model,
	persist_directory="../../resources/asset/chroma/2"
)
query = "哺乳动物"

"""
	相似性检索:  
		接收 字符串 或 向量
		filter 参数: dict， 支持过滤元数据
"""
doc_1 = db.similarity_search(query, k =3) # k，表示返回3个相关文档

doc_2 = db.similarity_search_by_vector(embeddings_model.embed_query(query), k=3)

doc_3 = db.similarity_search(
	query,
	k=3,
	filter={"source":"动物"}   # 针对文档中metadata进行过滤。 metadata字典中的source
)


"""
	通过 L2距离分数进行搜索
		score 范围 0 ～ 正无穷 。分值越小，检索到的文档越和提问的问题相似
"""
doc_4 = db.similarity_search_with_score(query, k=3)

"""
	通过余弦相似度分数进行检索
		score 范围  -1 ～ 1 。 值越接近1，检索到的文档越和问题相似
"""
doc_5 = db.similarity_search_with_relevance_scores(query, k=3)


"""
	MMR最大边际相关性
		MMR 是一种平衡「相关性」和「多样性」 的检索策略。 避免返回高度相似的冗余结果。
		lambda_mult：默认0.5，取值范围 0～1 ，用于确定结果多样性的程度。 0对应最大多样性。
"""
doc_6 = db.max_marginal_relevance_search(
	query,
	lambda_mult= 0.8
)