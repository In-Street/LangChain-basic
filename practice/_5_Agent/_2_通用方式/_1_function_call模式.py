"""
		create_tool_calling_agent:  依赖OpenAI 特定的功能调用格式，Qwen不支持，下面调用会报错： openai.BadRequestError: Error code: 400 - {'code': 20015, 'message': 'The parameter is invalid. Please check again.', 'data': None}

		之前在创建ChatOpenAI 实例时，加了参数 model_kwargs={"functions": True} ， 导致报上面错。去掉此参数后可正常运行

		运行后，输出信息显示： 直接就进行了工具调用 tavily_search_results_json 。而不会想 ReAct模式 先进行思考然后进行工具调用
"""
import dotenv, os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from practice.common_func import get_chat_model


dotenv.load_dotenv()  # 加载 tavily所需的 api_key
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

model = get_chat_model(max_tokens=300, temperature=0.0)

# 工具定义
search_tool = TavilySearchResults(max_results=2)  # 默认 name = tavily_search_results_json
tool_name = search_tool.name

# 模版定义
chat_prompt = ChatPromptTemplate.from_messages([
	("system", f"你是一个AI助手，根据用户提问，必要时调用{tool_name}工具，使用互联网检索数据"),
	("human", "{input}"),
	("placeholder","{agent_scratchpad}")  # 必须设置
])

agent = create_tool_calling_agent(
	llm=model,
	tools=[search_tool],
	prompt=chat_prompt
)

agent_executor = AgentExecutor(
	agent=agent,
	tools=[search_tool],
	verbose=True,
	handle_parsing_errors=True
)

response = agent_executor.invoke({"input": "国内公司大多使用的LangChain框架的哪个版本？此框架的使用相比于其他框架占比高吗？"})
print(response)

# 输出：
#  > Entering new AgentExecutor chain...
# Invoking: `tavily_search_results_json` with `{'query': '国内公司大多使用的LangChain框架的哪个版本？此框架的使用相比于其他框架占比高吗？'}`

# > Finished chain.
# {'input': '国内公司大多使用的LangChain框架的哪个版本？此框架的使用相比于其他框架占比高吗？', 'output': '根据现有信息，国内公司大多使用的LangChain框架版本是0.3.20。这个版本通过pip或conda安装，适用于仅使用框架的场景，不需要深入了解其源码构建过程。此外，源码安装方法也被推荐，因为它有助于深入理解框架的构建原理和详细机制。\n\n关于LangChain的使用占比，目前没有明确的数据表明它在所有开发框架中占比最高。然而，LangChain因其在处理大型语言模型（LLM）应用中的灵活性和强大的功能，被许多大厂和开发者广泛采用。这表明它在特定领域内具有较高的使用率，但具体占比还需更多数据支持。'}