"""
		远程提示词模版：https://smith.langchain.com/hub/hwchase17

		ReAct 提示词模版，包含 tools、tool_names、input、agent_scratchpad 变量 【https://smith.langchain.com/hub/hwchase17/react ，提供PromptTemplate 模版】

		使用方式：
			1. 将网址提供的文本内容复制，PromptTemplate.from_template("复制的内容") 来创建模版
			2. 使用 prompt = hub.pull('hwchase17/react')  创建模版
"""
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
import os,dotenv

from langchain_core.prompts import PromptTemplate

from practice.common_func import get_chat_model

dotenv.load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

model = get_chat_model(max_tokens=400,temperature=0)
#  工具定义
search_tool = TavilySearchResults(max_results=2)

# 模版定义, 使用LangChain Hub官方ReAct提示模版
prompt = hub.pull('hwchase17/react')

agent = create_react_agent(
	llm=model,
	tools=[search_tool],
	prompt=prompt,
)

agent_executor = AgentExecutor(
	agent=agent,
	tools=[search_tool],
	verbose=True,
)

response = agent_executor.invoke({"input": "查询一下Macbook Pro 16寸的最新款售价"})
print(response)

