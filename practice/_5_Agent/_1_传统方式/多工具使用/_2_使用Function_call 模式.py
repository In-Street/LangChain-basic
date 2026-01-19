"""
		function_call 模式：
			仅需修改 initialize_agent 的agent参数，agent = Agent.OPENAI_FUNCTIONS

			Agent.OPENAI_FUNCTIONS 仅对OpenAI 官方模型有效，Qwen无效
"""
import dotenv
import os
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

from practice.common_func import get_chat_model

dotenv.load_dotenv()  # 加载 tavily所需的 api_key
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

model = get_chat_model(max_tokens=500, temperature=0.0)

# 工具定义
search_tool = TavilySearchResults(max_results=2)

calculation = PythonREPL()
calc_tool = Tool(
	name='calculate',
	description='当且仅当获取到具体的数字数值后，必须调用此工具执行数学计算；用于计算股价涨幅、百分比变化、数值差值，仅输入纯数字计算公式，例如(300-200)/200*100，禁止输入百分号、中文、单位',
	func=calculation.run,
)


# 传统方式定义 AgentExecutor
#  AgentType.OPENAI_FUNCTIONS 对OpenAI 官方模型生效。Qwen模型无效
agent_executor = initialize_agent(
	llm=model,
	agent=AgentType.OPENAI_FUNCTIONS,
	tools=[search_tool, calc_tool],
	verbose=True,
	agent_kwargs={
		"prefix": "你必须严格按照以下规则执行：1. 所有问题必须先调用搜索工具获取准确数值，禁止凭记忆回答；2. 获取数值后必须调用计算工具执行运算；3. 禁止直接生成自然语言回答，必须完成工具调用后再整理结果。"
	},  # ✅ 重中之重！强制指令，源码级注入，强制模型调用工具
)

response = agent_executor.invoke('比亚迪2024年9月股价是多少？比去年2023年9月上涨了百分之几？')
print(response)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是专业的数据分析助手，先搜索获取数据，再进行计算，严格按步骤执行"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"), # 必加，Agent的核心占位符
# ])
#
# model_with_tools = model.bind_tools(
# 	tools=[search_tool, calc_tool], # 绑定工具列表，对Qwen模型生效
# 	tool_choice='auto'  # 模型自主判断「先调用哪个工具，后调用哪个工具」
# )
# agent = create_openai_tools_agent(llm=model_with_tools,tools=[search_tool,calc_tool],prompt=prompt)
# agent_executor = AgentExecutor(agent=agent, tools=[search_tool,calc_tool], verbose=True,max_iterations=3,handle_parsing_errors=True,return_only_outputs=True,)

# response = agent_executor.invoke('比亚迪2024年9月股价是多少？比去年2023年9月上涨了百分之几？')
# print(response)
