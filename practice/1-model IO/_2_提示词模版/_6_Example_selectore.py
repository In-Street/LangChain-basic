"""
	示例选择器：
			若 examples 示例组中给出大量示例 且 示例的相关性不同，比如：有天气示例、有数学运算示例 等等，当都丢给大模型后，大模型会消耗很多Token 但给出的结果又不准确。此时需要进行示例选择。

			选择策略：
					1. 语义相似选择： 通过余弦相似度等度量方式评估语义相关性，选择与输入问题最相似的 k 个示例。（需用向量数据库： Chroma【pip3 install chromadb】 、faiss-cpu 【pip3 install faiss-cpu】 ）
							余弦相似度： 通过计算两个向量的夹角余弦来衡量它们的相似性，取值范围：-1 ～ 1
								两个向量方向相同 = 1
								两个向量夹角为90° = 0
								两个向量完全相反 = -1

					2. 长度选择：根据输入的文本长度，从候选示例中筛选出长度最匹配的示例。比语义相似度计算更轻量，适合对响应速度要求高的场景

					3. 最大边际相关示例选择：优先选择与输入问题语义相似的示例，同时，通过惩罚机制避免返回同质化的内容（就是要多样化的示例，减少雷同）
"""
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings

# 语义相似选择策略举例

# 1. 定义嵌入模型:  利用嵌入模型将示例组内容转换为向量
embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002')

# 2. 定义示例组
examples_ = [
	{
		"question": "谁活得更久，穆罕默德·阿里还是艾伦·图灵?",
		"answer": """
			接下来还需要问什么问题吗？
			追问：穆罕默德·阿里去世时多大年纪？
			中间答案：穆罕默德·阿里去世时享年74岁。
		""",
	},
	{
		"question": "craigslist的创始人是什么时候出生的？",
		"answer": """
			接下来还需要问什么问题吗？
			追问：谁是craigslist的创始人？
			中级答案：Craigslist是由克雷格·纽马克创立的。
		""",
	},
	{
		"question": "谁是乔治·华盛顿的外祖父？",
		"answer": """
			接下来还需要问什么问题吗？
			追问：谁是乔治·华盛顿的母亲？
			中间答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。
		""",
	},
	{
		"question": "《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？",
		"answer": """
			接下来还需要问什么问题吗？
			追问：《大白鲨》的导演是谁？
			中级答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
		""",
	},
]

# 3. 定义示例选择器
examples_selector = SemanticSimilarityExampleSelector.from_examples(
	examples_,  # 可供选择的示例列表
	embeddings_model, # 嵌入模型，用于生成嵌入的嵌入类
	Chroma,  # 向量数据库，用于存储嵌入并进行相似性搜索的 VectorStore 类
	k=1  #  挑选出的示例数量
)

# 4. 选择与输入内容最相似的示例:  看问题的向量与示例组中哪个向量的夹角更小，从示例组中挑选出 示例选择器中定义的k 个示例
selected_examples = examples_selector.select_examples({'question': '玛丽·鲍尔·华盛顿的父亲是谁？'})
print(f'与输入内容相似的示例为：{selected_examples}')

# 5. 结合 少量样本示例的提示词模版 使用
prompt_template = PromptTemplate(
	template='question: {question}\nanswer: {answer}',
	input_variables=['question', 'answer'],
)
few_shot_prompt_template = FewShotPromptTemplate(
	examples_selector  = examples_selector,  # 示例选择器
	example_prompt=prompt_template,  # 示例模版
	suffix = 'question:{word} \n answer:',
	input_variables=['word']
)
prompt_value = few_shot_prompt_template.invoke({'word': 'xxx填写问题'})

# 6. 调用大模型
