"""
		1. pip3 install jq
			JSONLoader 内部使用 jq 来解析JSON文件。 jq 可对JSON格式的数据进行复杂的处理，包括：过滤、映射、转换、减少

		2. jq_schema 参数设定：依据提供的JSON文件格式， 提取所需字段，按层级结构表示

				.   :  表示提取所有字段

				.data.items[].title  : 表示 {"data":{"items":[{"title":"aaa"}]}} ，从 data 字典中取 items列表 中所有元素的title字段值

		3. text_content 参数设定
				Document 对象的 page_content 为str类型。 JSONLoader 加载文件后得到的是 dict类型，与str不匹配。通过 text_content = False ,将其转为JSON格式的字符串

"""
from langchain_community.document_loaders import JSONLoader

# 1.  加载文档中的所有数据
json_loader = JSONLoader(
	file_path="../../resources/asset/load/04-load.json",
	jq_schema=".",  # 必传参数， . 表示加载所有字段
	text_content=False # 默认获取到的为 dict类型，设置False 转成JSON格式的字符串，以匹配Document对象中 str类型的 page_content 属性
)

docs = json_loader.load()
print(docs,end="\n\n")


# 2. 加载04-load.json文件中，messages 列表里的所有content 字段
loader_2 = JSONLoader(
	file_path="../../resources/asset/load/04-load.json",
	jq_schema=".messages[].content"
)
docs_2 = loader_2.load()
print([c.page_content for c in docs_2],end="\n\n")

# 3. 加载 04-response.json 文件中，data 字典中 items列表中 的所有content字段
loader_3 = JSONLoader(
	file_path="../../resources/asset/load/04-response.json",
	jq_schema=".data.items[].content"
)
docs_3 = loader_3.load()
print([c.page_content for c in docs_3],end="\n\n")


# 4.  加载 04-response.json 文件中，data 字典中 items列表中的所有 title 字段和 content字段
loader_4 = JSONLoader(
	file_path="../../resources/asset/load/04-response.json",
	jq_schema=".data.items[]",
	content_key='.title +"  -> "+.content',  # 指定提取的字段
	is_content_key_jq_parsable=True  #  使用 jq解析 content_key
)
docs_4 = loader_4.load()
print([c.page_content for c in docs_4],end="\n\n")