"""
	实现方式：
			1. 在提示词模版中明确指定输出格式为XML
					chat_model.invoke( '请生成简短的记录，将影片结果附在 <movie> </movie>标签中 ' )

			2.  在模版变量中指定输出格式：
					PromptTemplate.from_template(
							template = "用户问题；{}，使用格式为：{ parser.get_format_instructions() }"
					)

					在模型调用完成后，返回的结果就是xml格式了。若将模型结果通过 XMLOutputParser # invoke 解析，那么结果为json格式了。
					就是说，XMLOutputParser 不会直接将模型的输出结果解析为原始XML字符串，而是会解析成字典。方便后续处理数据
"""
from langchain_core.output_parsers import XMLOutputParser

parser = XMLOutputParser()
instructions = parser.get_format_instructions()
print(instructions)
