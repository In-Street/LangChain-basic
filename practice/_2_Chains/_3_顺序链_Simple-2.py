"""
"""
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain_core.prompts import ChatPromptTemplate
from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=300)

template_1 = ChatPromptTemplate.from_template("为以下内容生成一个简洁的标题：{content}")  # 输入变量只有1个： content
llm_chain_1 = LLMChain(
	llm=model,
	prompt=template_1,
	output_key='title'  # 显式指定输出键为 title。默认为 text
)

# template_2 = ChatPromptTemplate.from_template("根据标题{input_aa}，为原内容写一段50字摘要：原内容：{content}")  # 错误写法，有两个输入变量。报错: Value error, Chains used in SimplePipeline should all have one input

template_2 = ChatPromptTemplate.from_template("根据标题{input_aa}，写一段50字的内容摘要")  #输入变量只有1个 input_aa，无需和上一条链的输出键值title一致
llm_chain_2 = LLMChain(
	llm=model,
	prompt=template_2,
	output_key='summary'  #显式指定输出键名
)

chain = SimpleSequentialChain(chains=[llm_chain_1, llm_chain_2], verbose=True)
response = chain.invoke({"input": "LangChain的Chain是核心组件，能串联多个模块实现复杂工作流，包含基础链、组合链等类型。"})
print(response)


