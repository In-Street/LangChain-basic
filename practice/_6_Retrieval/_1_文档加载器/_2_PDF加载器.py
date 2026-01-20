"""
		pip3 install pypdf
"""
from langchain_community.document_loaders import PyPDFLoader


# 加载本地文件
pdf_loader = PyPDFLoader(
	file_path='../../resources/asset/load/02-load.pdf'
)

docs_pdf = pdf_loader.load()
print(docs_pdf)


# 加载网络文件
pdf_loader_web = PyPDFLoader(
	file_path="https://arxiv.org/pdf/2302.03803"
)
docs_pdf_web = pdf_loader_web.load()
print(len(docs_pdf_web))
