from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from practice.common_func import get_chat_model


#  此处为单个文件。 多个文件时可设置文件目录
pdf_loader = PyPDFLoader('')

template = PromptTemplate.from_template(
	'对文字{text} 做简洁的总结：'
)

model = get_chat_model(max_tokens=300)

chain = LLMChain(
		llm=model,
		prompt=template
)

documents_chain = StuffDocumentsChain(
	llm_chain=chain,
	document_variable_name='text'
)

docs = pdf_loader.load()
res = documents_chain.invoke(input=docs)
print(res)
