"""
	ConversationKGMemory:
			基于 知识图谱 的对话记忆，不仅能存储实体，还能捕捉实体间复杂的关系，形成结构化的知识网络。

			知识图谱结构：将对话内容转化为 三元组形式 (头实体, 关系, 尾实体)

			pip3 install networkx
"""
from langchain_community.memory.kg import ConversationKGMemory
from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=100)

kg_memory = ConversationKGMemory(llm=model)

kg_memory.save_context({'input':"向Jay问好"},{"output":"Jay是谁"})
kg_memory.save_context({'input':"Jay是我的朋友"},{"output":"好的"})

variables = kg_memory.load_memory_variables({"input": "Jay是谁"})
print(variables)

kg_memory.get_knowledge_triplets("他最喜欢的歌曲是晴天")