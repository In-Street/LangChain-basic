"""
	此ReAct 模板未适配 Qwen 规范，导致模型无法解析，运行报错
"""

import dotenv
import os

from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from practice.common_func import get_chat_model

dotenv.load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

model = get_chat_model(max_tokens=400,temperature=0)
#  工具定义
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

# 模版定义，和网址提供的ReAct的PromptTemplate 模版内容一样，只是把 input和 agent_scratchpad 提取到单独的 human 和 system中
template_str = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

prompt = ChatPromptTemplate.from_messages([
	("system",template_str),
	("human","{input}"),
	("system","{agent_scratchpad}")
])


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

response = agent_executor.invoke({"input": "查询一下Macbook Pro 的最新款售价"})
print(response)

#   believing in you is the one thing that i have no trouble doing