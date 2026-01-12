"""
	StuffDocumentsChain 的LCEL 写法 ->  create_stuff_documents_chain
	将多个文档内容合并到单个prompt中，提供给LLM处理
"""
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=300)

template = PromptTemplate.from_template('基于文档{doc_}的内容，回答问题：{ques}')

chain = create_stuff_documents_chain(llm=model, prompt=template, document_variable_name='doc_')

doc_ = [
	Document(
		page_content='苹果，学名Malus pumila Mill.，别称西洋苹果、柰，属于蔷薇科苹果属的植物。'
		             '苹果是全球最广泛种植和销售的水果之一，具有悠久的栽培历史和广泛的分布范围。'
		             '苹果的原始种群主要起源于中亚的天山山脉附近，尤其是现代哈萨克斯坦的阿拉木图地区，提供了所有现代苹果品种的基因库。'
		             '苹果通过早期的贸易路线，如丝绸之路，从中亚向外扩散到全球各地。'
	),
	Document(
		page_content='香蕉是白色的水果，主要产自热带地区。'
	),
	Document(
		page_content='蓝莓是蓝色的浆果，含有抗氧化物质。'
	),
]

res = chain.invoke({'doc_': doc_, 'ques': '香蕉是什么颜色的？'})
print(res)
