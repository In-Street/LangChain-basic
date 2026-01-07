"""
	FewShotChatMessagePromptTemplate:
			 专门为聊天对话场景设计的少样本(few-shot) 提示词模版，继承自 BaseChatPromptTemplate。结合 ChatPromptTemplate 使用

			 特点：
			        1.  自动将示例格式化为聊天消息，HumanMessage、AIMessage
			        2. 输出结构化聊天消息，List[ BaseMessage ]
			        3.

"""
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from practice.common_func import get_chat_model


#  1. 实例化 FewShotChatMessagePromptTemplate ： 示例、示例模版
examples_ = [
	{'input': '2 🐦2', 'output': '4'},
	{'input': '2 🐦3', 'output': '8'},
]

chat_prompt_template = ChatPromptTemplate.from_messages(
	[
		('human', '{input} 的结果值是多少？'),
		('ai', '结果为：{output}')
	]
) # 依据上面的示例 examples 给出对应的变量模版

few_shot_chat_message_prompt_template = FewShotChatMessagePromptTemplate(
	example_prompt=chat_prompt_template,
	examples=examples_,
)

# 2. 将 FewShotChatMessagePromptTemplate 构造进 ChatPromptTemplate 中
final_template = ChatPromptTemplate.from_messages(
	messages=[
		('system','你是一个数学计算助手'),
		few_shot_chat_message_prompt_template, # 在第2节 ChatPromptTemplate 构造方法中的多类型参数时，提到支持 BaseChatPromptTemplate 类型
		('human','我提出一个问题，{question_input} 结果为多少？')
	]
)
prompt_value = final_template.invoke(input={'question_input':'3 🐦3'})
# print(prompt_value)


#3. 模型调用
model = get_chat_model(max_tokens=50)
# model_result = model.invoke(prompt_value)
# print(f'大模型输出：{model_result}')
# 大模型输出：content='我们来分析一下这个问题。\n\n你提到的表达式是：**3 🐦3**。\n\n看起来“🦜”是一个特殊的符号，可能代表某种数学运算或操作。根据你之前提供的例子：\n- **2 🐦2 = 4**\n- **2 🐦3 = 8**\n\n这些似乎不是常规的数学运算（如加、减、乘、除），但也不同于通常的指数运算（2^2 = 4，2^3 = 8），但看起来像是某种函数或规则。\n\n我们可以尝试找出这个“🦜”运算的规律。\n\n---\n\n### 分析已知的两个例子：\n\n1. **2 🐦2 = 4**\n2. **2 🐦3 = 8**\n\n观察这两个表达式，如果将“🦜”理解为某种函数，我们可能尝试猜测其意义。\n\n#### 假设“🦜”表示的是：\n- **指数运算**（2^2 = 4，2^3 = 8） → 成立\n- **乘法** → 2 × 2 = 4，2 × 3 = 6 → 不成立\n- **幂运算或其他组合运算** → 试着看看。\n\n如果我们假设“🦜”就是乘法，那么：\n- **3 × 3 = 9**\n\n但你已经知道“2 🐦2 = 4”，而 2 × 2 = 4，这个假设成立。如果你问“3 🐦3”，那很可能是 **3 × 3 = 9**。\n\n不过，你也可以考虑“🦜”代表某种特殊的“自定义运算”，前提是它是一致的。例如：\n- 如果“🦜”代表 **2^a × 2^b**，那：\n  - 2 🐦2 → 2^2 × 2^2 = 4 × 4 = 16（不对，与已知不一致）\n  - 所以应该不是这个。\n\n- 如果“🦜”代表 **2^(a × b)**，那么：\n  - 2 🐦2 → 2^(2×2) = 2^4 = 16（不对）\n\n- 如果“🦜”代表 **2 + 2 = 4**，**2 + 3 = 5**（但你给出的是 8，所以不成立）\n\n---\n\n### 最合理的猜测：\n既然你已知：\n- **2 🐦2 = 4**\n- **2 🐦3 = 8**\n\n那么设定这个特殊的“🦜”运算为某种规则：\n- **2 🐦2 = 2^2 = 4**\n- **2 🐦3 = 2^3 = 8**\n\n看起来 **“2 🐦x = 2^x”**。  \n如果你问 **3 🐦3**，那可能意味着：\n\n> **3 🐦3 = 3^3 = 27**\n\n---\n\n### ✅ 答案：**27**\n\n如果你能提供更多关于“🦜”的规则，我可以进一步验证是否一致。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 658, 'prompt_tokens': 84, 'total_tokens': 742, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': None}, 'model_name': 'Qwen/Qwen3-8B', 'system_fingerprint': '', 'id': '019b97a07a956d311673f0f53f1c6a0c', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None} id='run--4bce11ad-0b22-4e77-8a53-58e347f8a314-0' usage_metadata={'input_tokens': 84, 'output_tokens': 658, 'total_tokens': 742, 'input_token_details': {}, 'output_token_details': {'reasoning': 0}}

