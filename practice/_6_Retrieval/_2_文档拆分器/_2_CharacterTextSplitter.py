"""
		CharacterTextSplitter：
				separator 参数：
						''  :  空字符串，表示禁用分割符。 按照 chunk_size 分割，实际块长略小于chunk_size
						'。'  :  以句号分割。separator 优先原则，分割器首先以指定分割符分割，然后再考虑 chunk_size 。会出现块合并的现象，如下代码splitter_2
"""
from langchain_text_splitters import CharacterTextSplitter

#====================== 无分割符 ======================

text = """
		LangChain 是一个用于开发由语言模型驱动的应用程序的框架。他提供一套工具和抽象，使开发者能够更容易地构建复杂的应用程序
"""

splitter = CharacterTextSplitter(
	chunk_size=51,
	chunk_overlap=5,
	separator = ''  # 设为空字符串，表示禁用分割符
)

res = splitter.split_text(text)

for i, chunk in enumerate(res):
	print(f'块{i + 1}，长度：{len(chunk)}')
	print(chunk)
	print('-' * 50)
# 输出：
# 块1，长度：47
# LangChain 是一个用于开发由语言模型驱动的应用程序的框架。他提供一套工具和抽象，使开发
# --------------------------------------------------
# 块2，长度：21
# 象，使开发者能够更容易地构建复杂的应用程序


#====================== 指定分割符，先分割符 后chunk_size  ======================
text_2 = """
	这是一个示例文本啊。我们将使用CharacterTextSplitter将其分割成小块。分割基于字符数。
"""

splitter_2 = CharacterTextSplitter(
	# chunk_size=30,
	chunk_size=45,
	chunk_overlap=5,
	separator = '。'  # 句号分割符。 分割符优先
)

res_2 = splitter_2.split_text(text_2)

for i, chunk in enumerate(res_2):
	print(f'块{i + 1}，长度：{len(chunk)}')
	print(chunk)
	print('-' * 50)
# 	chunk_size=30 输出：
# 块1，长度：9
# 这是一个示例文本啊
# --------------------------------------------------
# 块2，长度：33
# 我们将使用CharacterTextSplitter将其分割成小块
# --------------------------------------------------
# 块3，长度：8
# 分割基于字符数。


# chunk_size=45 输出： 以句号分割后，若块长度相加比chunk_size小，会进行块合并，将上面chunk_size=30分割出的块1和块2 合并成一个块
# 块1，长度：43
# 这是一个示例文本啊。我们将使用CharacterTextSplitter将其分割成小块
# 块2，长度：8
# 分割基于字符数。


#====================== 指定分割符，chunk_overlap 如何生效  ======================

text_3 = """
	这是第一段文本。这是第二段内容。最后一段结束
"""

splitter_3 = CharacterTextSplitter(
	chunk_size=20,
	# chunk_overlap=5,
	chunk_overlap=8,
	separator='。',
	keep_separator=True
)
res_3 = splitter_3.split_text(text_3)

for i, chunk in enumerate(res_3):
	print(f'块{i + 1}，长度：{len(chunk)}')
	print(chunk)
	print('-' * 50)

# chunk_overlap=5 输出：chunk_overlap 无效，块2与块1并没有重叠内容
# 块1，长度：15
# 这是第一段文本。这是第二段内容
# --------------------------------------------------
# 块2，长度：7
# 。最后一段结束

# chunk_overlap=8 输出：此时 chunk_overlap 是生效的。以separator 分割后，「这是第二段内容。」的长度为8 与chunk_overlap 匹配，所以块2中能够显示出重叠内容。当chunk_overlap =5 时，分割器不会为了在块2中展示重叠内容而将「这是第二段内容。」截取，还是以 separator分割后的内容为主的，优先级高。
# 块1，长度：15
# 这是第一段文本。这是第二段内容
# --------------------------------------------------
# 块2，长度：15
# 。这是第二段内容。最后一段结束