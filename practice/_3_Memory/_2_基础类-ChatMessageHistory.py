"""
	ChatMessageHistory:  别名 InMemoryChatMessageHistory
		用于 存储和管理对话消息的基础类，直接操作消息对象，如：HumanMessage、AIMessage 。是其它记忆组件的底层存储工具。

"""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from practice.common_func import get_chat_model

# 1. 实例化
history = ChatMessageHistory()

# 2. 直接进行消息存储： HumanMessage 、 AIMessage
history.add_user_message('你好')

history.add_ai_message('很高兴认识你')

history.add_user_message('请列举Luke Combs 的播放量较高的10首歌曲，返回JSON格式')

# print(history.messages)


#3. 结合LLM
model = get_chat_model(max_tokens=200)
res = model.invoke(input=history.messages)
print(res)
