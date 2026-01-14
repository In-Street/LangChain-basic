"""
		load_memory_variables，返回 entities为空，并没有抽取出实体，因为LangChain 内置英文 Prompt 与 Qwen3-8B 中文模型不兼容 。
		需重新指定 提取提示词 和 生成摘要提示词，目前摘要生成不了

"""

from langchain.memory import ConversationEntityMemory
from langchain_core.prompts import PromptTemplate

from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=100, temperature=0)

# 自定义提示词，适配中文模型
ENTITY_EXTRACTOR_PROMPT_ZH = """从对话中提取所有重要实体名称。

对话历史：
{history}

最新对话：
{input}


要求：
1. 只返回实体名称
2. 每个实体用双引号包围
3. 用逗号分隔
4. 不要有其他文字

例如："实体1", "实体2", "实体3"

实体列表："""


ENTITY_SUMMARIZER_PROMPT_ZH ="""为实体"{entity}"生成或更新摘要。

对话上下文：
{history}
{input}

请为这个实体生成1-3句话的摘要，包含基本信息、在对话中的角色和相关重要信息。

描述："""


extraction_prompt = PromptTemplate(template=ENTITY_EXTRACTOR_PROMPT_ZH, input_variables=["history","input"])
summarization_prompt = PromptTemplate(template=ENTITY_SUMMARIZER_PROMPT_ZH, input_variables=["entity", "summary", "history", "input"], )

memory_ = ConversationEntityMemory(
	llm=model,
	input_key='input',
	output_key='output',
	return_messages=True,
	entity_extraction_prompt=extraction_prompt,  # 针对中文自定义提取模版
	entity_summarization_prompt=summarization_prompt # 针对中文，定义生成总结的模版
)

memory_.clear()
memory_.save_context({'input': '蜘蛛侠的好友包括钢铁侠、美队、冬兵、绿巨人'}, {'output': 'ok'})
memory_.save_context({'input': '蜘蛛侠住在纽约皇后区'}, {'output': 'ok'})
memory_.save_context({'input': '斯塔克工业为蜘蛛侠提供了战衣装备'}, {'output': 'ok'})


# memory_.clear()
# memory_.save_context({'input': "Spider-Man's friends include Iron Man, Captain America, the Winter Soldier, and the Hulk"}, {'output': 'ok'})
# memory_.save_context({'input': "Spider-Man lives in Queens, New York"}, {'output': 'ok'})
# memory_.save_context({'input': "Stark Industries provided Spider-Man with his suit and equipment"}, {'output': 'ok'})

print(memory_.load_memory_variables({'input': ''}))

print(memory_.entity_store.store)



# chain = LLMChain(
# 	llm=model,
# 	memory=memory_,
# 	prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE  # 使用LangChain提供的实体记忆模版
# )

# chain.invoke('蜘蛛侠的好友包括钢铁侠、美队、冬兵、绿巨人')
# chain.invoke('蜘蛛侠住在纽约皇后区')
# chain.invoke('斯塔克工业为蜘蛛侠提供了战衣装备')


# print(memory_.entity_store.store)

# entity_store = chain.memory.entity_store.entity_store_store
# print(entity_store)
