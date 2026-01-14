"""
	 ConversationBufferMemory:
	       1. 基础的对话记忆组件，专门用于按 原始顺序存储 完整的对话历史.
	            a. 消息存储： ConversationBufferMemory # save_context
	            b. 消息获取：
	                    ConversationBufferMemory # load_memory_variables({})  : dict ,  key值固定为 history

	                    ConversationBufferMemory # chat_memory.messages :  list[BaseMessage]

	       2. 场景： 对话轮次较少、依赖完整上下文的

	        3. 特点：
	                完整存储对话历史
	                无裁剪、无压缩
	                支持两种返回格式：
	                    return_messages = True:  返回 List[ BaseMessage ]
	                    return_messages = False:  默认，返回拼接的纯文本字符串

			4. 结合LLM
					LLMChain(llm=xx , template=xxx , memory=xxx)
					PromptTemplate:     仅在LLM #invoke 完后存储消息，首次invoke后，history为空串，存储的消息为拼接的字符串
					ChatPromptTemplate:  执行前存储用户输入 + 执行后存储输出，首次invoke调用后，history会显示完整的输入和输出消息，存储的消息为 list[BaseMessage]

"""
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from practice.common_func import get_chat_model


# 1. 创建 ConversationBufferMemory 实例
memory = ConversationBufferMemory()

# 2. 存储消息， inputs <-> 用户消息 、outputs <-> AI消息
#  默认使用 InMemoryChatMessageHistory # list[BaseMessage] 存储消息
memory.save_context(inputs={'human':'华语乐坛R&B代表性歌手有哪几位'},outputs={'ai':'Jay'})
memory.save_context(inputs={'输入':'他们的代表作有哪些'},outputs={'输出':'晴天'})

# 3. 返回所存储的消息，方式一 , key值固定为 history
msgs = memory.load_memory_variables({})
# print(msgs)  # dict类型， {'history': 'Human: 华语乐坛R&B代表性歌手有哪几位\nAI: Jay\nHuman: 他们的代表作有哪些\nAI: 晴天'}

# 3. 返回所存储的消息，方式二
message_list = memory.chat_memory.messages
# print(message_list) #  [HumanMessage(content='华语乐坛R&B代表性歌手有哪几位', additional_kwargs={}, response_metadata={}), AIMessage(content='Jay', additional_kwargs={}, response_metadata={}), HumanMessage(content='他们的代表作有哪些', additional_kwargs={}, response_metadata={}), AIMessage(content='晴天', additional_kwargs={}, response_metadata={})]


#4. 结合LLM -  使用 PromptTemplate
model = get_chat_model(max_tokens=100)
prompt_template = PromptTemplate.from_template('在1990～2000间，华语乐坛{question}代表性歌手有哪几位?列出3位，字数控制在100字内 ')

memory_prompt = ConversationBufferMemory(memory_key='history_for_prompy')  # 默认key值为： history

# 指定memory
llm_chain = LLMChain(llm=model, prompt=prompt_template, memory=memory_prompt)
# res = llm_chain.invoke(input={'question': 'R&B'})
# print(res)
# {'question': 'R&B', 'history_for_prompy': '', 'text': '1990～2000年间，华语乐坛R&B代表性歌手包括周杰伦、王力宏和陶喆。周杰伦以独特的音乐风格引领潮流，王力宏融合西方R&B与东方元素，陶喆则被誉为“华语R&B教父”，三人对华语R&B的发展有深远影响。'}

# res = llm_chain.invoke(input={'question': '在上面列出的5位歌手中，包含卫兰吗？'})
# print(res)
#输出：
# {'question': '在上面列出的5位歌手中，包含卫兰吗？', 'history_for_prompy': 'Human: R&B\nAI: 1990～2000年间，华语乐坛R&B代表性歌手包括周杰伦、王力宏和陶喆。周杰伦以独特的音乐风格引领潮流，王力宏融合西方R&B与东方元素，陶喆则被誉为“华语R&B教父”，三人对华语R&B的发展有深远影响。', 'text': '1990～2000年间，华语乐坛代表性歌手包括张学友、王菲、刘德华、周杰伦、陶喆。卫兰出生于1989年，其音乐事业主要在2000年后启动，因此不包含在1990～2000年间。'}


print('===============================================================')

# 4. 结合LLM - 使用 ChatPromptTemplate
chat_prompt_template = ChatPromptTemplate.from_messages([
	('system','你是一个常用知识查询助手'),
	MessagesPlaceholder(variable_name='history_for_chat_prompt'),  # 不能放在 ('human', xxx) 下面，否则 memory不会存储
	('human','回答一下：{ques}'),
])

memory_for_chat = ConversationBufferMemory(memory_key='history_for_chat_prompt', return_messages=True)

chain_for_chat = LLMChain(llm=model, prompt=chat_prompt_template, memory=memory_for_chat)
res_1 = chain_for_chat.invoke(input={'ques': '南宋首都设在哪个城市？只说明结果即可，无需过多阐述'})
print(res_1)
#{'ques': '南宋首都设在哪个城市？只说明结果即可，无需过多阐述', 'history_for_chat_prompt': [HumanMessage(content='南宋首都设在哪个城市？只说明结果即可，无需过多阐述', additional_kwargs={}, response_metadata={}), AIMessage(content='南宋首都设在临安。', additional_kwargs={}, response_metadata={})], 'text': '南宋首都设在临安。'}

res_2 = chain_for_chat.invoke(input={'ques': '刚刚问的是哪个朝代来着？'})
print(res_2)
# {'ques': '刚刚问的是哪个朝代来着？', 'history_for_chat_prompt': [HumanMessage(content='南宋首都设在哪个城市？只说明结果即可，无需过多阐述', additional_kwargs={}, response_metadata={}), AIMessage(content='南宋首都设在临安。', additional_kwargs={}, response_metadata={}), HumanMessage(content='刚刚问的是哪个朝代来着？', additional_kwargs={}, response_metadata={}), AIMessage(content='南宋。', additional_kwargs={}, response_metadata={})], 'text': '南宋。'}
