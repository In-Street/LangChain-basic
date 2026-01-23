"""

"""
import os, dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


dotenv.load_dotenv()

#  获取嵌入模型
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
embeddings_model = HuggingFaceEmbeddings(**embedding_config)

def main():
	documents = [
		Document(
			page_content="经济复苏：美国经济正在从疫情中强劲复苏，失业率降至历史低点。！",
		),
		Document(
			page_content="基础设施：政府将投资1万亿美元用于修复道路、桥梁和宽带网络。",
		),
		Document(
			page_content="气候变化：承诺到2030年将温室气体排放量减少50%。",
		),
		Document(
			page_content=" 医疗保健：降低处方药价格，扩大医疗保险覆盖范围。",
		),
		Document(
			page_content="教育：提供免费的社区大学教育。。",
		),
		Document(
			page_content="科技：增加对半导体产业的投资以减少对外国供应链的依赖。。",
		),
		Document(
			page_content="外交政策：继续支持乌克兰对抗俄罗斯的侵略。",
		),
		Document(
			page_content="枪支管制：呼吁国会通过更严格的枪支管制法律。",
		),
		Document(
			page_content="移民改革：提出全面的移民改革方案。",
		),
		Document(
			page_content="社会正义：承诺解决系统性种族歧视问题。",
		),
	]

	# 将文档向量化，存储到向量数据库中
	db = Chroma.from_documents(
		documents,
		embeddings_model,
		persist_directory="../../resources/asset/chroma/retriever-data"
	)
	print("数据文档向量化，存储到向量数据库，完成")

db_for_retriever = Chroma(
	embedding_function=embeddings_model,
	persist_directory="../../resources/asset/chroma/retriever-data"
)

if __name__ == "__main__":
	main()