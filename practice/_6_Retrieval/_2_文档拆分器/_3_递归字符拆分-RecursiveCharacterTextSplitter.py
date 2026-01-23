"""
	RecursiveCharacterTextSplitter :
			1. 以特定字符进行切割，提供切割的字符列表，默认    ["\n\n", "\n", " ", ""]

				首先以列表中第一个字符进行切割，若切块大于chunk_size，则以列表中第二个字符继续切割，以此类推。

			2.  特点：
					a. 保留上下文： 优先在自然语言边界处分割，减少信息碎片化。如段落、句子结尾
					b. 通过递归尝试多种分割符，将文本分割为大小接近chunk_size 的片段
					c. 适用于多种文本类型（代码、Markdown、普通文本），是LangChain 中最常用的分割器
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 示例1:  使用 create_documents() 切割
text = "LangChain框架特性\n\n多模型集成(GPT/Claude)\n记忆管理功能\n链式调用设计。文档分析场景示例：需要处理PDF/Word等格式。"
splitter = RecursiveCharacterTextSplitter(
	chunk_size=10,
	chunk_overlap=0,
	add_start_index=True
)
docs = splitter.create_documents([text])
for doc in docs:
	print(doc)
'''
	递归切割过程：
			第一阶段： \n\n顶级分割
				 块1->  LangChain框架特性   
				 块2-> 多模型集.... 后续所以内容
				 块1 和 块2 的长度都大于chunk_size ，需继续切割
			
			第二阶段：\n分割 	 
				块1: 无\n 符号，无空格符号，按字符分割，得到 ['L','a','n','g','C','h','a','i','n','框','架','特','性']
						前chunk_size个字符： LangChain框
						剩余字符： 架特性 
						
				块2: \n 分割，得到：  
						"多模型集成(GPT/Claude)", # 17字符  ，大于chunk_size 继续切割，无空格 以字符切分两部分： 多模型集成(GPT      /Claude)
						"记忆管理功能", # 6字符，无需切分
						"链式调用设计。文档分析场景示例：需要处理PDF/Word等格式。" # 36字符，大于chunk_size 继续切割，无空格 以字符切分
'''
# 	输出：
# page_content='LangChain框' metadata={'start_index': 0}
# page_content='架特性' metadata={'start_index': 10}
# page_content='多模型集成(GPT' metadata={'start_index': 15}
# page_content='/Claude)' metadata={'start_index': 24}
# page_content='记忆管理功能' metadata={'start_index': 33}
# page_content='链式调用设计。文档' metadata={'start_index': 40}
# page_content='分析场景示例：需要处' metadata={'start_index': 49}
# page_content='理PDF/Word等' metadata={'start_index': 59}
# page_content='格式。' metadata={'start_index': 69}


# 示例2:  使用 create_documents() 方法，将本地文件内容加载成字符串后进行切割
with open('../../resources/asset/load/08-ai.txt') as f:
	read_text = f.read()  # 返回字符串

splitter_2 = RecursiveCharacterTextSplitter(
	chunk_size=100,
	chunk_overlap=20,
)
doc_2 = splitter_2.create_documents([read_text])
for doc in doc_2:
	print(f'🔥{doc.page_content}')

# 示例3: 使用 split_documents() 方法，利用 PyPDFLoader 加载文档，对文档内容切割
pdf_loader = PyPDFLoader(file_path='../../resources/asset/load/02-load.pdf')
list_docs = pdf_loader.load()  # 返回 list[Document]

splitter_3 = RecursiveCharacterTextSplitter(
	chunk_size=200,
	chunk_overlap=0,
	length_function=len,
)

docs_3 = splitter_3.split_documents(list_docs)
for doc in docs_3:
	print(f'{doc.page_content}')