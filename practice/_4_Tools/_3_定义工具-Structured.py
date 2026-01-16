from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class Number(BaseModel):
	num1: int = Field(description='第一个整数')
	num2: int = Field(description='第二个整数')

def sub_numbers(num1, num2):
	return num1 - num2


function = StructuredTool.from_function(
	func=sub_numbers,
	description='',
	name='',
	args_schema=Number
)

# function.invoke({})