"""
	简单的将任何输入转换为字符串，从结果中提取 content 字段
"""
from langchain_core.output_parsers import StrOutputParser

from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=50)
response = model.invoke('')


#  获取字符串结果，方式一：调用属性
str_response = response.content

#  获取字符串结果，方式二： 利用字符串输出解析器
parser = StrOutputParser()
parse_response = parser.invoke(response)



