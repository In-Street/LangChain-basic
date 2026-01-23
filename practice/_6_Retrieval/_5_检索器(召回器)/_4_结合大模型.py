"""
		构建基础数据的向量数据库，把RAG 检索出的相关性文档 给到LLM作为上下文数据。让LLM给出回答
"""
import os
import dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter

from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=500)

# 1. 加载本地文档
loader = TextLoader("../../resources/asset/load/10-test_doc.txt")
load_docs = loader.load()

# 2. 文档拆分
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_chunks = splitter.split_documents(load_docs)

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
embeddings_model = HuggingFaceEmbeddings(**embedding_config)

# 3. 将拆分的文档存入向量数据库
db = Chroma.from_documents(doc_chunks, embeddings_model, persist_directory="../../resources/asset/chroma/retriever-llm")


# 4. 创建提示词模版
template = """请使用以下提供的文本内容来回答问题。仅使用提供的文本信息，如果文本中
没有相关信息，请回答"抱歉，提供的文本中没有这个信息"。
文本内容：
{context}
问题：{question}
回答：
"
"""
prompt = PromptTemplate.from_template(template)

# 5. 根据问题检索文档
question = "列举一些北京有名建筑"
retriever = db.as_retriever(search_kwargs={"k": 3}) # 返回文档数为3
retriever_docs = retriever.invoke(question)

# 6. 将检索出的相关文档作为上下文，回答问题
chain = prompt | model
llm_res = chain.invoke({"question": question, "context": retriever_docs})
print(llm_res.content)
