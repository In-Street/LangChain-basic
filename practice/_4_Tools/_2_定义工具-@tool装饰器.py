"""
		@tool 装饰器
			默认使用函数名称作为工具名称，可通过 name_or_callable 来覆盖；
			装饰器使用函数的「文档字符串」 作为「工具的描述」，因此函数必须提供文档字符串

		Tool常用属性：
			 name ： str ， 在提供给LLM或Agent的工具集中必须是唯一的
			description：str，描述工具的功能，LLM或Agent将使用此描述作为上下文，来确定工具的使用
			args_schema： Pydantic BaseModel , 用于提供更多信息或验证预期参数
			return_direct:  bool，仅对Agent。 True时，调用工具后Agent将停止并将结果直接返回给用户
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class Number(BaseModel):
	num1: int = Field(description='第一个整数')
	num2: int = Field(description='第二个整数')

#  1. 工具定义
@tool(
	name_or_callable='两数相加',
	description='计算两个整数的和',
	return_direct=True,
	args_schema= Number
)
def add_numbers(num1, num2):
    return num1 + num2

print(f'{add_numbers.name}')   # 默认为函数名。 name_or_callable 覆盖
print(f'{add_numbers.description }') # 函数说明信息
print(f'{add_numbers.args}') # 参数说明 。 BaseModel 类型

#2. 工具调用

res = add_numbers.invoke({'num1': 10, 'num2': 20})
print(res)
