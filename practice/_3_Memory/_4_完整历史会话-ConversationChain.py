"""
		ConversationChain:
			1. LLMChain 的子类，提供默认的 prompt 和 memory

					prompt:  为PromptTemplate ，两个变量为 input 、history

					memory:  为ConversationBufferMemory

			2. 简化使用 LLMChain + ConversationBufferMemory + PromptTemplate
"""
from langchain.chains.conversation.base import ConversationChain

from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=100)

chain = ConversationChain(llm=model)
res = chain.invoke({'input': '北宋的首都是哪个城市？对应到现在是哪个城市？直接给出结果，无需过多描述'})
print(res,end='\n\n')
# {'input': '北宋的首都是哪个城市？对应到现在是哪个城市？直接给出结果，无需过多描述', 'history': '', 'response': '北宋的首都是东京，对应现在是中国的开封市。'}


res_1 = chain.invoke({'input': '上次我提问的是哪个朝代？直接给出结果，无需过多描述'})
print(res_1)
# {'input': '上次我提问的是哪个朝代？直接给出结果，无需过多描述', 'history': 'Human: 北宋的首都是哪个城市？对应到现在是哪个城市？直接给出结果，无需过多描述\nAI: 北宋的首都是东京，对应现在是中国的开封市。', 'response': '北宋。'}