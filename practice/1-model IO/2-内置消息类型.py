"""
	1. langchain内置消息类型：

			SystemMessage:
					设定AI行为规则、背景信息、角色。确保回复符合预期风格，提升特定领域的回答质量。如：客服机器人、专业咨询、特定风格输出；
					在简单的问答、通用对话时，SystemMessage设置与否差距不大；

			HumanMessage:

			AIMessage:

			ChatMessage: 可自定义角色的通用消息类型

			FunctionMessage / ToolMessage:  函数调用/工具调用 场景下使用

"""
import os

from langchain_openai import ChatOpenAI
import dotenv

dotenv.load_dotenv()

chat_model = ChatOpenAI(model=os.getenv('ZHIPUAI_MODEL_NAME'),
                        api_key=os.getenv('ZHIPUAI_API_KEY'),
                        base_url=os.getenv('ZHIPUAI_API_BASE'),
                        max_tokens=150
                        )