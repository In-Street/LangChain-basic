"""
	1. str - format：占位符字符串模版

	2. 不同类型的提示词模版：
			a. PromptTemplate: LLM提示模版，用于生成字符串提示。使用python的字符串模版来提示

			b. ChatPromptTemplate:  聊天提示模版，用于组合各种角色的消息模版，传入聊天模型

			c. FewShotPromptTemplate:  样本提示词模版，通过示例来教模型如何回答

			d. XxxMessagePromptTemplate:  如SystemMessagePromptTemplate、HumanMessagePromptTemplate、AIMessagePromptTemplate

			f. PipelinePromptTemplate: 管道提示词模版，用于把几个提示词组合一起使用
"""

# 1. 位置参数方式
str_a = '歌曲：{0}，艺人：{1}'.format('一首歌的时间', '周杰伦')
print(str_a)

# 2. 带有关键字参数方式
str_b = '歌曲：{song}，艺人：{artist}'.format(song='晴天', artist='周杰伦')
print(str_b)

# 3. 字典解包方式
dit_a = {'song': '枫', 'artist': '周杰伦'}
str_c = '歌曲：{song}，艺人：{artist}'.format(**dit_a)
print(str_c)
