from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

from practice.common_func import get_chat_model

"""
	LLMChain: 
			至少包括一个 提示词模版、一个语言模型。 使用与单次问答，无上下文、无记忆场景
"""

model = get_chat_model(max_tokens=50)

chat_prompt_template = ChatPromptTemplate.from_messages([
	('system', ''),
	('human', '回答问题:{question}')
])

#  配置任务链
llm_chain = LLMChain(
	llm=model,  # 调用的语言模型
	prompt=chat_prompt_template,  # BasePromptTemplate 类型
	verbose=True  # True: 显示执行过程中的详细日志
)

# 执行任务链
response = llm_chain.invoke(input={'question': ' Jay 的代表做有哪些'})
print(response)
