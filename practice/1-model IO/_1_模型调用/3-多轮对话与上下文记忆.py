"""
	大模型本身是没有上下文记忆能力的

"""
from langchain_core.messages import SystemMessage, HumanMessage

from practice.common_func import get_chat_model

chat_model = get_chat_model()


chat_model.invoke([
	SystemMessage(content='你的名字叫 Jay'),
	HumanMessage(content=''),
])

# 当再次发起调用时，大模型是不知道它叫 Jay 的
chat_model.invoke([
	HumanMessage(content='你叫什么名字'),
])



############## 一次性发起时  ##############

chat_model.invoke([
	SystemMessage(content='你的名字叫 Jay'),
	HumanMessage(content='问题A'),
	HumanMessage(content='你叫什么名字')  # 此时输出的结果大模型是知道自己名字的
])