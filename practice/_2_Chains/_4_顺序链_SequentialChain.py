"""
	input1 、input2 -> LLMChain  -> LLMChain -> output1、output2

	1. 多变量支持：  允许不同子链有独立的输入/输出变量

	2. 映射：
			需显式定义 变量如何从一个链传递到下一个链。精准的命名输入关键字和输出关键字，来明确链之间的关系

	3. 流程控制：
			支持分支、条件逻辑

"""
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=500)

# 任务链1:   翻译成中文
template_1 = PromptTemplate.from_template('把下列内容翻译成中文：{content}')
chain_1 = LLMChain(
	llm=model,
	prompt=template_1,
	output_key='transaction_content'
)

# 任务链2: 对翻译后的内容生成摘要
template_2 = PromptTemplate.from_template('把下列内容：{transaction_content}，生成摘要')
chain_2 = LLMChain(
	llm=model,
	prompt=template_2,
	output_key='summary'
)

# 任务链3: 根据生成的摘要，识别语言类型
template_3 = PromptTemplate.from_template('下列内容：{summary}，使用的是什么语言')
chain_3 = LLMChain(
	llm=model,
	prompt=template_3,
	output_key='language_type'
)

# 任务链4:  对2中生成的摘要，以3中识别出的语言类型，生成评论
template_4 = PromptTemplate.from_template('针对内容：{summary}，使用语言类型为：{language_type}, 作出评论')
chain_4 = LLMChain(
	llm=model,
	prompt=template_3,
	output_key='comments'
)

full_chain = SequentialChain(
	chains=[chain_1, chain_2, chain_3, chain_4],
	verbose=True,
	input_variables=['content'],
	output_variables=['transaction_content','summary',  'comments']  # 需要几个输出内容就填写几个
)

response = full_chain.invoke(input={'content': 'Do you think, because I am poor, obscure, plain, and little, I am soulless and heartless? You think wrong! — I have as much soul as you — and full as much heart! And if God had gifted me with some beauty and much wealth, I should have made it as hard for you to leave me, as it is now for me to leave you. I am not talking to you now through the medium of custom, conventionalities, nor even of mortal flesh: it is my spirit that addresses your spirit.'})
print(response)
