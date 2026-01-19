"""
	1. initialize_agent() 方法中针对传入的agent枚举，会转为对应的class
			AGENT_TO_CLASS 定义了对应关系。 CONVERSATIONAL_REACT_DESCRIPTION -> ConversationalAgent ,
			传统方式不用显式定义模版，其默认的模版为： ConversationalAgent # create_prompt ， 变量有 input_variables = ["input", "chat_history", "agent_scratchpad"]，最后返回 PromptTemplate

	2. handle_parsing_errors
			a. 当使用ReAct模式时，要求LLM的响应必须遵守严格的格式 ，如 Thought:  Action:  Observation:  。当LLM返回了自由文本，导致AgentOutputParser解析器无法识别，报错
			b.  handle_parsing_errors 参数用于控制Agent在解析工具调用或输出发生错误时进行容错。当设置为 True：
					自动捕获错误并修复： 当Agent解析失败，Agent不会直接崩溃，而是将错误信息传递给LLM，让LLM自行修正并重试
					降级处理： 若重试后仍失败，Agent会返回一个友好的错误消息，如 i couldn't process that request ，而不是抛出异常

					设置为False，Agent解析失败会直接抛出异常，适用开发调试阶段快速发现问题

"""
import dotenv, os
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import TavilySearchResults

from practice.common_func import get_chat_model

dotenv.load_dotenv()  # 加载 tavily所需的 api_key
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

model = get_chat_model(max_tokens=100, temperature=0)

# 获取搜索工具实例
search = TavilySearchResults(max_results=1)  # 带有默认的name、description。 也可使用下面 Tool 方式，自定义的description

# 获取记忆实例
memory = ConversationBufferMemory(
	memory_key="chat_history"   # 此处需指定与默认模版中变量相同的名称
)

# 获取 agent、AgentExecutor 实例。每个枚举都有相对应的类，可参考 langchain.agents.types # AGENT_TO_CLASS
agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION

agent_executor = initialize_agent(
	agent=agent,
	tools=[search],
	llm=model,
	verbose=True,
	memory =memory,
	handle_parsing_errors=True
)

res_1 = agent_executor.invoke("最新款Macbook Pro 是哪个型号，其芯片是M5吗？售价多少？")
print(res_1)
# 输出：
# {'input': '最新款Macbook Pro 是哪个型号，其芯片是M5吗？售价多少？', 'chat_history': '', 'output': '最新款的 MacBook Pro 是 14 英寸型号，搭载了 M5 芯片。其起售价为人民币 12,999 元，教育优惠起售价为人民币 12,249 元。该型号于 2025 年 10 月 22 日正式发售，提供深空黑色和银色两种外观选项。'}

print('======================')

res_2 = agent_executor.invoke("iPhone呢？")
print(res_2)
# 输出： 带有 chat_history
# {'input': 'iPhone呢？', 'chat_history': 'Human: 最新款Macbook Pro 是哪个型号，其芯片是M5吗？售价多少？\nAI: 最新款的 MacBook Pro 是 14 英寸型号，搭载了 M5 芯片。其起售价为人民币 12,999 元，教育优惠起售价为人民币 12,249 元。该型号于 2025 年 10 月 22 日正式发售，提供深空黑色和银色两种外观选项。', 'output': '您是想问最新款的 iPhone 是哪个型号吗？如果是，那么最新款的 iPhone 是 iPhone 15 系列，包括 iPhone 15、iPhone 15 Plus、iPhone 15 Pro 和 iPhone 15 Pro Max。这些型号搭载了 A17 Pro 芯片，其中 Pro 系列还配备了灵动岛设计和 USB-C 接口。具体售价会根据不同的型号和存储容量有所不同，通常起售价在人民币 5,999 元左右。如果您有更具体的问题，欢迎继续提问！'}
