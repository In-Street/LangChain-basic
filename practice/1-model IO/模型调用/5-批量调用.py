from langchain_core.messages import SystemMessage, HumanMessage

from practice.common_func import get_chat_model

"""
	使用 batch 方法进行批量调用，返回列表
"""
chat_model = get_chat_model()

message_1=[SystemMessage(content=''),HumanMessage(content='')]
message_2=[SystemMessage(content=''),HumanMessage(content='')]


#返回结果是一个列表， 针对batch方法传入的列表参数，每个元素（message_1、message_2）都会返回一个AIMessage
batch_response = chat_model.batch([message_1, message_2])

print(batch_response)
