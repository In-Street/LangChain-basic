"""
		pip3 install unstructured
		pip3 install markdown

		mode 参数：
				single: 默认 , 返回单个大文档
				elements： 将Markdown文档按语义元素（标题、段落、列表、表格等）拆分成多个独立的小文档（ Element 对象），而不是返回单个大文档

"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader

md_loader = UnstructuredMarkdownLoader(
	file_path="asset/load/06-load.md",
	strategy="fast"
)
docs = md_loader.load()
print(len(docs))

for doc in docs:
	print(doc)
