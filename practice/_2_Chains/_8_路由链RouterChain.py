"""
		1. 两条子链，分别负责退款、发货

		2. 路由模版，仅输出关键词

		3. 路由决策链

		4. 封装组件

		注意点：
			1.   LLMRouterChain 绑定的路由模版，必须设置输出解析器RouterOutputParser【问题：Value error, LLMRouterChain requires base llm_chain prompt to have an output parser that converts LLM text output to a dictionary with keys 'destination' and 'next_inputs'. 】
					在路由模版中指明：将模型输出的结果转换为JSON字符串，键包括：destination、next_inputs，value值必须为str类型

			2.  每条子链的output_key 不能自定义，统一为 text。否则MultiPromptChain # invoke 时报错：ValueError: Missing some output keys: {'text'}

			3. 默认链模版不能因为是固定统一回复而不设定("human", "{input}") ，否则路由到默认链执行时，会报错。默认链通过system来固定统一回复话术，("ai","xxx) 指定的并不是LLM最后回答的内容。

"""
from langchain.chains.llm import LLMChain
from langchain.chains.router import LLMRouterChain, MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=200)

# 1. 定义两条子链，分别负责退款业务、发货业务
refund_template = ChatPromptTemplate.from_messages([
	("system", "回答用户提出的退款问题，语气温和，说明退款流程：1. 订单页申请 2. 上传凭证 3. 24小时到账"),
	("human", "{input}")
])

refund_chain = LLMChain(
	llm=model,
	prompt=refund_template,
	# output_key="refund_reply"  # 不能自定义，统一使用默认的text
)

delivery_template = ChatPromptTemplate.from_messages([
	("system", "回答用户提出的发货问题，语气温和，说明发货时效：48小时内，超时补偿5元优惠券"),
	("human", "{input}")
])

#  默认链，路由无匹配时触发
delivery_chain = LLMChain(
	llm=model,
	prompt=delivery_template,
	# output_key="delivery_reply"
)


default_template = ChatPromptTemplate.from_messages([  # 默认链也需指定input变量，否则报错
	("system","仅回答：抱歉亲，我暂未理解你的问题，可咨询退款/发货相关问题哦～，不要添加任何额外的文本"),
	("human", "{input}"),
])

default_chain = LLMChain(
	llm=model,
	prompt=default_template,
	# output_key="default_reply"
)

# 2. 路由模版，仅输出关键字
router_str = """
		请严格按照以下规则处理用户输入，仅输出纯JSON字符串，无其他任何内容：
		1. JSON必须包含2个键：destination、next_inputs，且两个键的值均为纯字符串；
		2. destination：路由关键词，仅可选以下值之一：refund（退款问题）、delivery（发货问题）、other（其他问题）；
		3. next_inputs：直接赋值为用户的原始输入内容，保持字符串原样，不做任何修改；
		4. 仅输出JSON，不要添加解释、示例、备注等任何额外文本！

		用户输入：{input}
"""
router_output_parser = RouterOutputParser()
router_template = PromptTemplate.from_template(
	template=router_str,
	output_parser=router_output_parser, # 必须配置专属输出解析器，将大模型的纯文本输出转换为包含destination（路由目标关键词）和next_inputs（传递给子链的输入参数）的字典
)

# 3. 初始化路由决策链。路由模版使用ChatPromptTemplate时，在LLMChain中指定输出解析器还是会报错：LLMRouterChain requires base llm_chain prompt to have an output parser。所以直接使用PromptTemplate创建模版，直接指定输出解析器
base_router_chain = LLMChain(
	llm=model,
	prompt=router_template,
	verbose=True
)

router_chain = LLMRouterChain(
	llm_chain=base_router_chain,
	verbose=True
)

# 4.  初始化MultiPromptChain，封装所有组件
multi_prompt_chain = MultiPromptChain(
	router_chain=router_chain,
	default_chain=default_chain,
	destination_chains={  # 路由模版中关键词 映射到 子链
		"refund": refund_chain,
		"delivery": delivery_chain,
		"other": default_chain
	},
	verbose=True
)

# 5. 退款场景
response_refund = multi_prompt_chain.invoke({"input": "我买的显示器有瑕疵，想进行退款"}) # 调用invoke，那么每条子链的output_key 必须为text
print(response_refund)
# 输出：
# {'input': '我买的显示器有瑕疵，想进行退款', 'text': '您好，很抱歉听到您购买的显示器有瑕疵，这确实会影响使用体验。我们非常重视您的反馈，并会尽快为您处理退款事宜。\n\n请您按照以下流程进行操作：\n\n1. **订单页申请**：请登录我们的平台，进入您对应的订单页面，找到“售后服务”或“申请退款”的入口，提交退款申请。\n2. **上传凭证**：在申请过程中，请上传相关证明材料（如商品照片、检测报告或与瑕疵相关的说明），以便我们更好地评估情况。\n3. **退款处理**：我们将在收到您的申请和凭证后，在**24小时内**完成审核，并将退款金额返还至您的原支付账户。\n\n如果您在操作过程中有任何疑问，或需要帮助，欢迎随时联系我们的客服，我们将竭诚为您提供支持。希望尽快帮您解决问题，祝您生活愉快！'}

# 5. 发货场景
response_delivery = multi_prompt_chain.invoke({"input": "都5天了，什么时候发货啊"})
print(response_delivery)
# 输出：
# {'input': '都5天了，什么时候发货啊', 'text': '您好，非常感谢您的耐心等待！\n\n关于发货时效，我们承诺会在您下单后 **48小时内发货**。由于近期订单量较大，可能会有一些延迟，给您带来了不便，我们深表歉意。\n\n目前您的订单已超过48小时，我们正在加急处理中，预计将在 **1-2个工作日内** 完成发货。为了表达我们的歉意，我们为您准备了 **5元的优惠券**，如果您的订单还在等待中，我们将优先为您补偿，并尽快安排发货。\n\n请您放心，我们会尽快完成发货，如有任何进展，也会第一时间通知您。感谢您的理解与支持，祝您生活愉快！'}

# 5. 关键词无匹配
response_default = multi_prompt_chain.invoke({"input": "不想发货的话，我想直接联系你领导，他的电话是多少？"})
print(response_default)
# 输出：
# {'input': '不想发货的话，我想直接联系你领导，他的电话是多少？', 'text': '抱歉亲，我暂未理解你的问题，可咨询退款/发货相关问题哦～'}