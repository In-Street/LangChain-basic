"""
	ConversationSummaryBufferMemory:
			1. LangChain 提供的一种 混合记忆机制 。 结合了 完整对话记录的ConversationBufferMemory  和 摘要记忆的ConversationSummaryMemory。
				在保留 最近对话原始记录 的同时，对较早的对话内容进行 智能摘要。

			2. 特点：
					a. 保留最近N条对话的原始记录 ：  确保了最新交互的完整上下文

					b.  较早历史记录摘要处理： 对超出缓冲区的旧对话进行压缩，避免信息过载

					c.平衡细节与效率： 既不会丢失关键细节，又能处理长对话

			3.  在新对话输入，
						缓冲区未满（未达到设置的max_token_limit） -> 存入原始对话
						缓冲区已满 -> 触发智能压缩，把早期的对话内容做总结压缩，用摘要替换旧对话。 保留了「最新的完整对话」 + 「早期对话的总结内容」
				最终传给大模型的永远是 token数 <= max_token_limit 的上下文，既保证上下文的连贯，又不会超限

			4. Qwen3-8B 模型，没有实现token计算的 get_num_tokens_from_messages 方法，此方法是 OpenAI用来计算多轮对话message的token数。

"""
from langchain.memory import ConversationSummaryBufferMemory
from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=100)

memory = ConversationSummaryBufferMemory(
	llm=model,
	max_token_limit= 40,
	return_messages=True
)

memory.save_context(inputs={'in': '在1990～2000间，华语乐坛R&B代表性歌手有哪几位?'},
                    outputs={'out': '周杰伦、王力宏、陶喆'})

memory.save_context(inputs={'in': '在1990～2000间，华语乐坛粤语代表性歌手有哪几位?'},
                    outputs={'out': '刘德华、伍佰、谭咏麟'})

variables = memory.load_memory_variables({})
print(variables)
