"""
	日期解析器：

"""
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import ChatPromptTemplate

from practice.common_func import get_chat_model

chat_model = get_chat_model()

chat_prompt = ChatPromptTemplate.from_messages([
	("system", "{format_instructions}"),  # 在 system 指定输出格式
	("human", "{request}")
])
output_parser = DatetimeOutputParser()

chain = chat_prompt | chat_model | output_parser
resp = chain.invoke({"request": "中华人民共和国是什么时候成立的",
                     "format_instructions": output_parser.get_format_instructions()})
print(resp)
print(type(resp))  # datetime 类型
