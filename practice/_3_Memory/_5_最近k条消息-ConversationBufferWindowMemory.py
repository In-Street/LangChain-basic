"""
	 在之前使用的 ConversationBufferMemory ，它可以无限的将历史对话信息填充到 history 中，来给LLM提供上下文的背景。引发的问题包括：占用内存量高、消耗token多、响应变慢，况且LLM本身存在最大输入的Token限制。
	 过于久远的对话数据往往并不能对当前轮次的问答提供有效的信息。

	 ConversationBufferWindowMemory:
	    只会保存最近k个对话

	    特点：
	        适合长对话场景、与Chain/models 无缝集成、通过 return_messages = True /False 控制返回格式
"""
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

from practice.common_func import get_chat_model


# 手动存入消息
memory = ConversationBufferWindowMemory(k=2)

memory.save_context(inputs={'输入': '销量高且口碑好的B级轿车有什么品牌？'}, outputs={'输出': '不知道'})
memory.save_context(inputs={'输入': '销量高且口碑好的A级轿车有什么品牌？'}, outputs={'输出': '不知道'})
memory.save_context(inputs={'输入': '销量高且口碑好的C级轿车有什么品牌？'}, outputs={'输出': '不知道'})

print(memory.load_memory_variables({}),end='\n\n')  # 只输出最后两条
print(memory.chat_memory.messages,end='\n\n') # 输出3条全部消息
print(memory.buffer_as_messages) # 只输出最后两条


#  结合LLM
model = get_chat_model(max_tokens=100)

memory_ = ConversationBufferWindowMemory(k=3)

template = PromptTemplate.from_template('提问问题：{ques}')

chain = LLMChain(llm=model, prompt=template, memory=memory_)

chain.invoke({'ques':'我计划购买B级类型的轿车，请提供三个品牌。直接回答，无需描述'})
chain.invoke({'ques':'大众、丰田、别克三个品牌各属于哪个国家？直接回答，无需描述'})
chain.invoke({'ques':'Luke Combs 是什么类型歌手？直接回答，无需描述'})
res = chain.invoke({'ques': 'A级、B级、C级，我计划够买其中的哪个类型的的轿车？知道的话直接给出答案，不知道的话就返回不知道就可以了'})
print(res)
# k=2 时，结果为不知道。  {'ques': '我计划够买A级、B级、C级中的哪个类型的的轿车？知道的话直接给出答案，不知道的话就返回不知道就可以了', 'history': 'Human: 大众、丰田、别克三个品牌各属于哪个国家？直接回答，无需描述\nAI: 大众：德国  \n丰田：日本  \n别克：美国\nHuman: Luke Combs 是什么类型歌手？直接回答，无需描述\nAI: Luke Combs 是乡村歌手。', 'text': '不知道。'}
# k=3时，  貌似也是不知道
