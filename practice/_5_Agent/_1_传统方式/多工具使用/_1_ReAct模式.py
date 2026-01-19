"""
		PythonREPL : LangChain 封装的工具类，可进行数学计算
		pip3 install langchain-experimental
"""
import dotenv,os
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from practice.common_func import get_chat_model


dotenv.load_dotenv()  # 加载 tavily所需的 api_key
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

model = get_chat_model(max_tokens=100)

# 1. 定义搜索工具
search_tool = TavilySearchResults(max_results=2)

# 2. 定义计算工具
calculation = PythonREPL()
calc_tool = Tool(
	name='Calculation',
	description='用于执行数学计算',
	func = calculation.run,
)

# 3. 创建 AgentExecutor 实例
agent_executor = initialize_agent(
	llm = model,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # ReAct 模式
	tools=[search_tool, calc_tool],
	verbose=True
)

# 4.  invoke 获取结果
response = agent_executor.invoke('比亚迪2025年12月31日股价是多少？比去年2024年12月31日上涨了百分之几？')
print(response)
