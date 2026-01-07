"""

			FewShotPromptTemplate： 少量样本示例的提示词模版，每个示例是一个字典
					1. 结合 PromptTemplate 使用，此模版用于格式化提供的示例
					2. 示例列表，元素字典类型
					3. 构造参数：
							example_prompt :  PromptTemplate类型，模版变量与examples的示例相对应
							examples： 示例
							suffix：
							input_variables:

			FewShotChatMessagePromptTemplate:   结合 ChatPromptTemplate 使用

			Example selectors（示例选择器）：
"""
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from practice.common_func import get_chat_model

# 创建 PromptTemplate实例。 模版的变量名和下面的示例有对应关系
prompt_template = PromptTemplate.from_template(template='输入：{input} , 输出：{output}')

# 提供示例
examples = [
	{'input': '北京天气怎么样？', 'output': '北京市'},
	{'input': '南京下雨吗？', 'output': '南京市'},
	{'input': '武汉热吗？', 'output': '武汉市'},
]

few_shot_pt = FewShotPromptTemplate(
	example_prompt=prompt_template,  # PromptTemplate 类型，用于格式化独特的示例
	examples=examples,
	suffix='input_suf:{input_param}，output_suf:',  # 放在示例之后的提示词模版
	input_variables=['input_param']  # 传入 suffix 模版中的变量名
)

# 如下面打印的一长串内容，将一长串内容全部给到大模型，大模型根据内容中提供的示例，来回答提出的问题
prompt_value = few_shot_pt.invoke(input={'input_param':'太原正在下雪吗？'})
print(prompt_value)  #  text='输入：北京天气怎么样？ , 输出：北京市\n\n输入：南京下雨吗？ , 输出：南京市\n\n输入：武汉热吗？ , 输出：武汉市\n\ninput_suf:太原正在下雪吗？，output_suf:'

# 调用LLM，根据示例，结果应该会输出城市名，太原市
model = get_chat_model(max_tokens=100)
model_invoke_response = model.invoke(prompt_value)
print(f'LLM结果值：{model_invoke_response}')
