"""
	ChatPromptTemplate:
			1.  实例化方式：
					a. 使用构造方法：传入参数messages ，可设为元组列表 [  (角色, 内容) , (角色, 内容)  ]
					b. 类方法： from_message( messages=[] )，也是传入参数 messages
					c. 无论哪种方式，参数messages为列表，元素类型包括：
							str、元组（常用此方式）、字典、消息类型（BaseMessage）、Chat提示词模版类型（BaseChatPromptTemplate）、消息提示词模版类型（BaseMessagePromptTemplate）

			2. 调用提示词模版方式：
					invoke(input={'name': '小智', 'question': '周杰伦的代表作品有哪些}):  返回类型 ChatPromptValue

					format(name=xx, question=xx)):  返回类型 str

					format_message( name=xx, question=xx) ): 返回类型 list

					format_prompt( name=xx, question=xx) ): 返回类型 ChatPromptValue

			3. 丰富的实例化参数：

			4. 结合LLM

			5. 插入消息列表 MessagesPlaceholder
					当 ChatPromptTemplate 模版中的消息类型、个数不确定的时候，可以使用 MessagesPlaceholder(variable_name=''变量名 ) 占位，
					在 template.invoke() 时，去赋值变量名为列表
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
	MessagesPlaceholder
from practice.common_func import get_chat_model

# 1.  实例化方式：使用构造方法。元组类型
template_by_init = ChatPromptTemplate(
	messages=[
		('system', '你是一个AI助手，你的名字是{name}'),
		('human', '我的问题是{question}')
	]
)

# 1.  实例化方式：使用构造方法。，messages列表元素：字典类型、提示词模版类型、消息类型、消息提示词模版类型
template_by_init_2 = ChatPromptTemplate(
	messages=[
		{'role': 'system', 'content': '你是一个AI助手，你的名字是{name}'},  # 字典类型
		{'role': 'human', 'content': '我的问题是{question}'},

		ChatPromptTemplate.from_messages([('system', '你是一个AI助手，你的名字是{name}')]),  # 提示词模版类型BaseChatPromptTemplate

		SystemMessage(content='你是一个AI助手，你的名字是{name}'),  # 消息类型 BaseMessage 时，若有变量 name，后续invoke也不会给此变量赋值。使用此类型时，不能在放入变量了
		HumanMessage(content='我的问题是{question}'),

		SystemMessagePromptTemplate.from_template('你是一个专家{name}'),  # 消息提示词模版类型时，可以设置变量。 和上面消息类型的区别
		HumanMessagePromptTemplate.from_template('我的问题是{question}')

	]
)

# 1. 实例化方式： 类方法
template_by_classmethod = ChatPromptTemplate.from_messages(
	messages=[('system', '你是一个AI助手，你的名字是{name}'), ('human', '我的问题是{question}')])

#  2. 调用提示词模版方式： invoke
prompt_value_by_invoke = template_by_init.invoke(input={'name': '小智', 'question': '周杰伦的代表作品有哪些？'})  #

print(type(prompt_value_by_invoke))  # ChatPromptValue
print(prompt_value_by_invoke.messages)  # 列表，包含两个元素： SystemMessage、 HumanMessage

# 2. 调用提示词模版方式: format()
by_format = template_by_init.format(name='小智A', question='包括《晴天》吗？')
print(type(by_format))  # str
print(f'调用提示词模版-format： {by_format}')

# 2. 调用提示词模版方式： format_messages()
by_format_message = template_by_init.format_messages(name='小智B', question='包括《枫》吗？')
print(type(by_format_message))  # list
print(f'调用提示词模版-format_messages： {by_format_message}')  # 列表，包含两个元素： SystemMessage、 HumanMessage

# 2. 调用提示词模版方式： format_prompt()
by_format_prompt = template_by_init.format_prompt(name='小智C', question='包括《说好的幸福呢》吗？')
print(type(by_format_prompt))  # ChatPromptValue
print(f'调用提示词模版-format_prompt： {by_format_prompt}')
print(
	f'ChatPromptValue 转为消息构成的列表：{by_format_prompt.to_messages()}，ChatPromptValue 转为字符串：{by_format_prompt.to_string()}')

# 4. 结合 LLM
model = get_chat_model(max_tokens=30)
# invoke_result = model.invoke(prompt_value_by_invoke)  # 传入 ChatPromptValue

# 5. 插入消息列表 MessagesPlaceholder： 在创建模版时，不确定消息类型、消息个数，用MessagesPlaceholder代替。 在invoke时进行列表赋值，可赋值为多个、多种类型
cpt = ChatPromptTemplate.from_messages(messages=[
	('system', '你是AI助手，{name}'),
	MessagesPlaceholder(variable_name='msgs')
])

invoke_res = cpt.invoke({'name': 'AA',
                         'msgs': [HumanMessage(content='周杰伦的代表作品有哪些？'),
                                  AIMessage(content='代表作有：晴天、一路向北')]})  # 可赋值为多个元素、多种类型
print(f'插入消息列表: {invoke_res}')
