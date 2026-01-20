"""
	1. 不同类型文档对应的加载器：
			TextLoader
			PyPDFLoader
			CSVLoader
			JSONLoader
			UnstructuredHTMLLoader
			UnstructuredMarkdownLoader
			DirectoryLoader :  文件目录

	2.   xxxLoader 实例， load() 方法都会返回 list[Document]
"""
from langchain_community.document_loaders import TextLoader

# 1. 获取加载器实例
text_loader = TextLoader(
	file_path='../../resources/asset/load/01-langchain-utf-8.txt',
	encoding='utf-8',  #  要与文件编码方式一致，否则报错
)

# 返回 Document对象列表。  对象属性： metadata 元数据、page_content 内容
docs = text_loader.load()
print(docs)
# 输出：
# [Document(metadata={'source': '../../resources/asset/load/01-langchain-utf-8.txt'}, page_content='LangChain 是一个用于构建基于大语言模型（LLM）应用的开发框架，旨在帮助开发者更高效地集成、管理和增强大语言模型的能力，构建端到端的应用程序。它提供了一套模块化工具和接口，支持从简单的文本生成到复杂的多步骤推理任务')]
