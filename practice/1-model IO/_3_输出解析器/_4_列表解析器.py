"""
	CommaSeparatedListOutputParser:
		可将大模型结果转换为 逗号分隔的列表，list[str]

	实现方式： 与前面 json格式一样
"""
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
instructions = parser.get_format_instructions()
print(instructions)

invoke_result = parser.invoke('周杰伦，王力宏，林俊杰')  # 中文逗号
print(invoke_result) # ['周杰伦，王力宏，林俊杰']
print(len(invoke_result)) # 1


invoke_result_2 = parser.invoke('周杰伦,王力宏,林俊杰')  # 英文逗号
print(invoke_result_2) # ['周杰伦', '王力宏', '林俊杰']
print(len(invoke_result_2)) # 3