"""
		VectorStoreRetrieverMemory:

				基于向量检索的记忆机制，将历史对话存储在向量数据库中
				减少历史对话中的冗余信息：根据新输入内容，通过语义相似度检索出历史相关信息，作为上下文给到LLM，减少不相关的历史对话。每次调用时，会查找关联最高的k个文档

				场景：适合需长期记忆和语义理解的复杂对话系统
"""
from langchain.memory import VectorStoreRetrieverMemory, ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 2.定义ConversationBufferMemory对象
memory = ConversationBufferMemory()
memory.save_context({"input": "我最喜欢的食物是披萨"}, {"output": "很高兴知道"})
memory.save_context({"Human": "我喜欢的运动是跑步"}, {"AI": "好的,我知道了"})
memory.save_context({"Human": "我最喜欢的运动是足球"}, {"AI": "好的,我知道了"})

# 3.定义向量嵌入模型
embeddings_model = OpenAIEmbeddings(
	model="text-embedding-ada-002"
)

# 4.初始化向量数据库
vectorstore = FAISS.from_texts(memory.buffer.split("\n"), embeddings_model)  # 空初始化

# 5.定义检索对象
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))

# 6.初始化VectorStoreRetrieverMemory
memory = VectorStoreRetrieverMemory(retriever=retriever)
print(memory.load_memory_variables({"prompt": "我最喜欢的食物是"}))
