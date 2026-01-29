"""
    比较 指定system角色与不指定时，LLM回复的可控性

            system 角色： 设定了模型的身份、回复规则、边界。能让模型的输出更可控、更符合预期。
"""
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=100)


# 大概率会回答出 Java相关的问题
template = PromptTemplate.from_template(
	template="你是一个Python编程助手，不需要回答其他编程语言的问题。我的问题是{question}",
)

chain_1 = template | model
res = chain_1.invoke({"question":"Java语言中基本类型有哪些？"})
print(type(res))
print(f'纯文本交互结果：{res}')

print("===" *40)


#  不会回答Java相关的问题，严格遵守system角色约束
chat_prompt_template = ChatPromptTemplate.from_messages(
	[
		("system","你是只回答Python问题的助手，不需要回答其他编程语言的问题 "),
		("human","{question}"),
	]
)

chain_2 = chat_prompt_template | model
res_2 = chain_2.invoke({"question": "Java语言中基本类型有哪些？"})
print(type(res_2))
print(f'明确指定system规则，结果：{res_2}')
