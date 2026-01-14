"""
	ConversationTokenBufferMemory：
			是LangChain 中基于 Token 数量控制的对话记忆机制。
			若字符数超过指定数目，它会切掉此对话的早期部分，保留与最近对话相对应的字符数量。

"""
from langchain.memory import ConversationTokenBufferMemory
from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=50)

memory = ConversationTokenBufferMemory(
	llm=model,   # 此处需要指定大模型，因为需要利用LLM来计算 Token
	max_token_limit=20  # 按Token数计算是否存储时，有可能只存了输出，不会像之前存储类以输入/输出一对一对的去存储
)

memory.save_context(inputs={'in':'R&B'},outputs={'out':'周杰伦'})
memory.save_context(inputs={'in':'乡村'},outputs={'out':'Luke Combs'})

print(memory.load_memory_variables({}))

#  Qwen3-8B 模型没有实现 get_num_tokens_from_messages()，所以无法计算消息的token数，此存储类还不能进行使用