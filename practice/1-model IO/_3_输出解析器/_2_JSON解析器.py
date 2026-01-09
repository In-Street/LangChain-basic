"""
	适用于严格结构化输出的场景：API调用、数据存储、下游任务处理

	实现方式：
			1.  使用方通过提示词指明返回json格式，将LLM返回结果直接通过 JsonOutputParse 解析，parse.invoke( LLM返回的结果 )

			2. 使用 JsonOutputParse # get_format_instructions()  方法返回字符串" Return a JSON object" 。 在提示词中告诉LLM返回结果时，可直接调用该方法，将返回值作为模版变量值
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from practice.common_func import get_chat_model

# model = get_chat_model(max_tokens=50)
model = None

chat_prompt_template = ChatPromptTemplate.from_messages(
	[
		('system', '你的角色是{role}'),
		('human', '{question}'),
	]
)
parser = JsonOutputParser()

#  方式一： 提示词中直接指明返回 JSON格式
prompt_value_chat = chat_prompt_template.invoke(
	input={'role': 'AI助手', 'question': '用中文讲一个简短的戏笑话。返回JSON格式的数据'})  #  若此处没有告诉LLM返回json格式，后续直接用 JsonOutputParse#invoke时，会报错

model_response = model.invoke(prompt_value_chat)
print(type(model_response))  # AIMessage
#
json_result = parser.invoke(model_response)
print(type(json_result))  # dict
print(json_result)

# 方式二： get_format_instructions
prompt_template = PromptTemplate.from_template(template='回答用户的查询，满足的格式为{format}，问题为{questions}',
                                               partial_variables={"format": parser.get_format_instructions()})

prompt_value = prompt_template.invoke({'questions': '想出一个简短的笑话'})
model_response = model.invoke(prompt_value)

json_response = parser.invoke(model_response)
print(json_response) #   {'joke': '为什么电脑会感冒？因为它打开了太多窗口。'}
print(type(json_response)) # dict

print(parser.get_format_instructions())  # Return a JSON object


#  管道符结构使用： 上述调用过程为： 模版 # invoke  -> 结果值传入 LLM # invoke  -> 结果值传入 输出解析器 # invoke ， 每次将上一步invoke的结果传入下一步invoke ，此时可简化为管道符结构。有严格的先后顺序
chain = prompt_template | model | parser
result = chain.invoke({'questions': '想出一个简短的笑话'}) # 参数为第一个调用invoke时的参数
print(result)
