"""
	tavily.com -> 创建API密钥
	使用 tavily 搜索工具
"""
from langchain.agents import AgentType, Agent, initialize_agent
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import Tool
import dotenv, os
from practice.common_func import get_chat_model



dotenv.load_dotenv()  # 加载 tavily所需的 api_key
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

model = get_chat_model(max_tokens=100)

# 获取搜索工具实例
search = TavilySearchResults(max_results=2) # 带有默认的name、description。 也可使用下面 Tool 方式，自定义的description

# 转为Tool
# search_tool = Tool(
# 	name='实时搜索工具',
# 	description='用于检索互联网上的信息',
# 	func=search.run
# )

# 获取Agent 实例: ReAct模式
agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION

# 获取Agent 实例: Function_call 模式
# agent_function_call = AgentType.OPENAI_FUNCTIONS

# 获取 AgentExecutor 实例
agent_executor = initialize_agent(
	tools=[search],
	agent=agent,
	llm=model,
	verbose=True,
)

response = agent_executor.invoke('最新款Macbook Pro 是哪个型号？售价多少？')
print(response)
# 输出： 此输出格式使用的默认的提示词模版，当使用通用方式时，需要自定义模版
#  > Entering new AgentExecutor chain...
# 我需要查询北京今天的天气情况。
# Action: tavily_search_results_json
# Action Input: 北京今天天气怎么样
# Observation: xxxxx
# Thought:通过搜索结果，可以得知北京今天（假设当前日期是2026年1月18日）的天气是小雪，气温为-11°C到-3°C，风力为东北风1级，空气质量良好。

# Final Answer: 北京今天有小雪，气温在-11°C到-3°C之间，风力为东北风1级，空气质量优。

# > Finished chain.
# {'input': '北京今天天气怎么样', 'output': '北京今天有小雪，气温在-11°C到-3°C之间，风力为东北风1级，空气质量优。'}
