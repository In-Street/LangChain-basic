"""

"""
import json
from typing import Optional

from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from practice.common_func import get_chat_model, get_model

"""
    LLM 根据提问内容，是可以分析出要调用哪个工具。
    模型返回的结果中会包含 tool_calls 字段，里面就是「是否调用工具、调用哪个工具、传什么参数」；
"""
@tool
def move_file(file_name:Optional[str] = 'A.txt' , target_path: Optional[str] = '桌面') -> str:
	"""当用户需要移动文件时调用此工具"""
	return f'成功将文件{file_name}，移动到{target_path}'

model = get_chat_model(max_tokens=50,temperature=0)

# 工具列表，自定义工具
# tools = [move_file]
# tools_model = model.bind_tools(tools)  #无Agent，纯LLM时，必须给模型绑定工具列表，返回一个「带工具的新模型实例，调用此实例的invoke」  是invoke前传入tool的核心方法

# 工具列表，LangChain 官方封装好的成品工具，注意此处为实例
tools_model = model.bind_tools([MoveFileTool()])

# LLM调用
response = tools_model.invoke(
	input=[HumanMessage('将当前目录下的llm.txt移动/Users/chengyufei/Downloads 文件夹下 ')],
	# input=[HumanMessage('北京明天天气怎么样')],
)
print(response)
# 当提问：将文件未命名2.txt移动到桌面，LLM分析出需要调用的工具后，输出选择的工具信息包括名称、参数，content为空：
#绑定自定义工具时输出 content='' additional_kwargs={'tool_calls': [{'id': '019bc14716fa613331caed5f60951fd6', 'function': {'arguments': '{"file_name": "未命名2.txt", "target_path": "桌面"}', 'name': 'move_file'}, 'type': 'function'}]
#绑定官方工具时输出  content='' additional_kwargs={'tool_calls': [{'id': '019bc16e58374e3e681cd1b8a5c48e0c', 'function': {'arguments': '{"source_path": "未命名2.txt", "destination_path": "/Users/your_username/Desktop"}', 'name': 'move_file'}, 'type': 'function'}], 'refusal': None}

# 当提问：北京明天天气怎么样，LLM分析出不需要调用对应的工具，content不为空，不包含工具调用的信息：
# content='我目前无法提供实时天气信息，建议您使用天气预报应用或访问气象网站查询北京明天的天气情况。' additional_kwargs={'refusal': None}


"""
	LLM分析出使用哪个工具后，无Agent情况下进行工具调用
"""
if 'tool_calls' in response.additional_kwargs:
	function_dict = response.additional_kwargs['tool_calls'][0]['function']
	arguments_ = json.loads(function_dict['arguments'])  #注意要将提取出的arguments字符串转为json
	file_tool = MoveFileTool()
	run_result = file_tool.run(arguments_)
	print(f'工具调用结果：{run_result}')  #输出：  File moved successfully from llm.txt to /Users/chengyufei/Downloads.

"""
	按照硅基文档提供的方式调用，LLM可以选择工具
"""
client = get_model(max_tokens=50, temperature=0)
def get_weather(city: str):
	return f'{city}天气很好！！！'

tools=[
{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a given city.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                        'description': 'The name of the city to query weather for.',
                    },
                },
                'required': ['city'],
            },
        }
    }
]

# response = client.chat.completions.create(
# 		model='Qwen/Qwen3-8B',
# 		max_tokens=50,
#         messages=[{
#         'role': 'user',
#         'content': '笔记本电脑有哪些高端品牌',
#     }],
#         temperature=0.01,
#         top_p=0.95,
#         stream=False,
#         tools=tools
#     )
# print(response)
# 当提问：北京明天天气怎么样时。在 tool_calls 中显示调用的工具。LLM会根据description选择相应的工具
# 输出：choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content='', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='019bc1344cc3059abae0eec7cfb659fd', function=Function(arguments='{"city": "北京"}', name='get_weather'), type='function')]))]


# 当提问：笔记本电脑有哪些高端品牌。tool_calls 为None
# 输出：  choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessage(content='高端笔记本电脑品牌通常指的是那些在设计、性能、质量和品牌声誉方面都表现出色的制造商。以下是一些知名的高端笔记本电脑品牌：\n\n1. **Apple (苹果)** - 以其MacBook系列著称，包括MacBook', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))]