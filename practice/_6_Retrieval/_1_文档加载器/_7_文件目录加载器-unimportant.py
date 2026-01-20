"""
	pip3 install unstructured

"""
from langchain_community.document_loaders import DirectoryLoader, PythonLoader


# 定义DirectoryLoader对象,指定要加载的文件夹路径、要加载的文件类型和是否使用多线程
directory_loader = DirectoryLoader(
	path="../../resources/asset/load",
	glob="*.py",
	use_multithreading=True,
	show_progress=True,
	loader_cls=PythonLoader
)

docs = directory_loader.load()

print(len(docs))
for doc in docs:
	print(doc)
