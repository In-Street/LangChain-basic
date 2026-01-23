"""
	TokenTextSplitter:  按Token数量分割

			依据： Token数量+自然语言边界。在严格按照Token数据进行切割的同时，后优先在自然边界处（如句号）切断，以保证语义的完整性。

"""
import tiktoken
from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter

#  方式一
text = "人工智能是一个强大的开发框架。它支持多种语言模型和工具链。人工智能是指通过计算机程序模拟人类智能的一门科学。自20世纪50年代诞生以来，人工智能经历了多次起伏。"
splitter = TokenTextSplitter(
	chunk_size=33,  # 每块最大 token数
	chunk_overlap=0,  # 重叠token数为0
	encoding_name='cl100k_base'  # 使用OpenAI 编码器，将文本转换为token序列
)

texts = splitter.split_text(text)

for i, chunk in enumerate(texts):
	print(f'块{i + 1}，长度:{len(chunk)}，内容：{chunk}')
	print('-' * 30)


# 方式二：使用 CharacterTextSplitter
splitter_2 = CharacterTextSplitter.from_tiktoken_encoder(
	encoding_name='cl100k_base',
	# chunk_size=18,
	chunk_size=33,
	chunk_overlap=0,
	separator='。',
)
docs = splitter_2.split_text(text)

# 初始化tiktoken编码器
encoding = tiktoken.get_encoding('cl100k_base')

for i, doc in enumerate(docs):
	token_ids = encoding.encode(doc) # 编码文本，获取token证书列表，len() 取Token数量
	print(f'块{i+1}，长度：{len(token_ids)}，内容：{doc}')

	#  解码token整数，得到每个token对应的文本
	# token_texts = [encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
	#                for token_id in token_ids]
	# print(f'Token对应的文本：{token_texts}')
	print('-' * 30)

# tiktoken.encoding_for_model('')  根据模型获取对应的编码，避免手动记编码名称