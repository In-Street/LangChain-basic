"""
		之前对话信息的存储，全部保存的话太过浪费，截断存储时无论按照 条数限制 还是token限制 无法保证既节省内存又保证对话质量。

		ConversationSummaryMemory :
			LangChain 中 智能压缩对话历史 的记忆机制，通过LLM自动生成对话内容的摘要，而不是存储原始对话文本。
"""
from langchain.memory import ConversationSummaryMemory
from langchain_core.chat_history import InMemoryChatMessageHistory

from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=100)

# 1. 实例化 ConversationSummaryMemory 时，若没有历史消息，采用构造方法进行实例化
memory = ConversationSummaryMemory(
	llm=model,
)

memory.save_context(inputs={'in': '在1990～2000间，华语乐坛R&B代表性歌手有哪几位?'},
                    outputs={'out': '周杰伦、王力宏、陶喆'})

memory.save_context(inputs={'in': '在1990～2000间，华语乐坛粤语代表性歌手有哪几位?'},
                    outputs={'out': '刘德华、伍佰、谭咏麟'})

#  输出的内容是LLM总结过后的
dict_ = memory.load_memory_variables({})
# print(dict_,end='\n\n\n')
# {'history': 'In the 1990s to 2000s, the representative R&B singers in the Mandarin pop music scene include Jay Chou, Wang Le-hong, and Tao Zhe, while the representative Cantonese singers in the same period include Liu Dehua, Wu Bai, and Tam Kam-lin.'}


# 2.  若已存在历史消息，可调用from_messages 来创建 ConversationSummaryMemory 实例
history = InMemoryChatMessageHistory()
history.add_user_message('周杰伦的代表作有：晴天、枫、说好的幸福呢、一首歌的时间')

summary_memory = ConversationSummaryMemory.from_messages(llm=model,
                                                         chat_memory=history)
print(summary_memory.load_memory_variables({}), end='\n\n\n')


# 当存在新的输入时，会根据 旧摘要 + 新输入 ，重新生成新的摘要
summary_memory.save_context(inputs={'in': '王力宏的代表作有哪些？'}, outputs={'out': '唯一、爱错'})
dict_2 = summary_memory.load_memory_variables({})
print(dict_2)

#  存储了历史交互的消息
messages = summary_memory.chat_memory.messages
