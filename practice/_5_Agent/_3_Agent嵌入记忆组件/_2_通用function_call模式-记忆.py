"""

"""
import dotenv,os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from practice.common_func import get_chat_model



dotenv.load_dotenv()  # 加载 tavily所需的 api_key
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

model = get_chat_model(max_tokens=100, temperature=0)

# 获取搜索工具实例
search = TavilySearchResults(max_results=1)  # 带有默认的name、description
tools = [search]


# 记忆实例
memory = ConversationBufferMemory(
	return_messages=True,
	memory_key="chat_history"
)

prompt_template = ChatPromptTemplate.from_messages(
	[
		("system",f"你是一个AI助手，根据用户提问，必要时调用{search.name}工具，使用互联网检索数据"),
		("system","{chat_history}"),  # 添加 chat_history 变量，用于记录上下文记忆
		("placeholder","{agent_scratchpad}"),
		("human","{input}")
	]
)

# Agent 、AgentExecutor 实例
agent = create_tool_calling_agent(
	llm=model,
	tools=tools,
	prompt=prompt_template
)

agent_executor = AgentExecutor(
	agent=agent,
	memory=memory,
	tools=tools,
	verbose=True,
)

res_1 = agent_executor.invoke({"input":"Macbook Pro 使用年限通常是多久？"})
print(res_1)
# 输出： 直接进行工具调用，无需像 ReAct模式先进行思考
# > Entering new AgentExecutor chain...
# Invoking: `tavily_search_results_json` with `{'query': 'Macbook Pro 使用年限通常是多久？'}`
# > Finished chain.
# {'input': 'Macbook Pro 使用年限通常是多久？', 'chat_history': [HumanMessage(content='Macbook Pro 使用年限通常是多久？', additional_kwargs={}, response_metadata={}), AIMessage(content='根据苹果官方的环境责任报告，全新MacBook Pro的“一手使用时间”大约为4年。然而，大多数MacBook Pro的实际使用寿命远超过这个时间，因为它们通常会被转卖、转让或送回苹果供其他人使用。此外，分析师Horace Dediu在2018年的研究中提到，苹果设备的平均寿命约为4.25年，但这一数据包含了所有苹果产品，因此对MacBook Pro的参考价值有限。\n\n从硬件角度来看，MacBook Pro的固态硬盘（SSD）性能和存储容量在发布时就已经确定，无法自行升级。因此，硬件的过时速度可能会影响其使用寿命。不过，由于MacBook Pro的硬件通常较为耐用，且软件兼容性良好，许多用户在使用几年后仍能继续使用它们。\n\n综上所述，MacBook Pro的使用年限通常在4年左右，但实际使用时间可能更长，具体取决于用户的使用习惯、软件需求以及硬件维护情况。', additional_kwargs={}, response_metadata={})], 'output': '根据苹果官方的环境责任报告，全新MacBook Pro的“一手使用时间”大约为4年。然而，大多数MacBook Pro的实际使用寿命远超过这个时间，因为它们通常会被转卖、转让或送回苹果供其他人使用。此外，分析师Horace Dediu在2018年的研究中提到，苹果设备的平均寿命约为4.25年，但这一数据包含了所有苹果产品，因此对MacBook Pro的参考价值有限。\n\n从硬件角度来看，MacBook Pro的固态硬盘（SSD）性能和存储容量在发布时就已经确定，无法自行升级。因此，硬件的过时速度可能会影响其使用寿命。不过，由于MacBook Pro的硬件通常较为耐用，且软件兼容性良好，许多用户在使用几年后仍能继续使用它们。\n\n综上所述，MacBook Pro的使用年限通常在4年左右，但实际使用时间可能更长，具体取决于用户的使用习惯、软件需求以及硬件维护情况。'}

print('======================')

res_2 = agent_executor.invoke({"input":"iPhone呢？"})
print(res_2)
# 输出： 根据记忆，知道此问题问的是 iPhone的使用年限，而不是其他方面
# > Entering new AgentExecutor chain...
# Invoking: `tavily_search_results_json` with `{'query': 'iPhone 使用年限通常是多久？'}`