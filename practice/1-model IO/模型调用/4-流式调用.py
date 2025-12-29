"""
	模型调用：
		Runnable协议，聊天模型、提示词模版、输出解析器、检索器、智能体 均实现了该协议
		Runnable 公共方法：
			invoke: 单条输入，等待LLM生成完整响应，然后再一次性返回调用结果。

			stream: 流式响应，逐个token的实时返回响应结果。 返回结果是可迭代对象

			batch: 处理批量输入

		异步方法：
			astream:
			ainvoke:
			abatch:
			astream_log: 异步流式返回中间步骤，及最终响应
			astream_events: 异步流式返回链中发生的事件
"""
from langchain_core.messages import SystemMessage, HumanMessage

from practice.common_func import get_chat_model



chat_model = get_chat_model(max_tokens=100, streaming=True)

# stream 方法进行调用
invoke_response = chat_model.stream([
	SystemMessage(content='你是一个大模型应用开发领域的专家'),
	HumanMessage(content='帮我指定一份从Java开发转大模型应用开发的计划')])

# 循环输出每块内容
for chunk in invoke_response:
	print(chunk.content, end='', flush=True)

print('\n结束')
