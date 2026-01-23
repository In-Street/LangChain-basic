"""
		SemanticChunker:
			根据文本的语义结构进行分块，使每个分块保持语义的完整性，从而提高检索增强的效果

			参数：
				breakpoint_threshold_type: 定义语义边界的检测算法，决定分块的时机。取值如下：
						percentile：计算相邻句子嵌入向量的余弦距离，取距离分布的第N百分位值作为阈值，高于此值则分割。
											适用于 常规文本（文章、报告）

						standard_deviation ： 以均值 + N倍标准差为阈值，识别语义突变点
															适用于 语义变化剧烈的文档(如技术手册)

						interquartile： 用四分位距(IQR) 定义异常值边界，超过则分割
												适用于 长文档(如书籍)

						gradient：基于嵌入向量变化的梯度检测分割点(需自定义实现）
										适用于 实验性需求

				breakpoint_threshold_amount： 断点阈值。 控制分割的粒度，值越小分割越细块越多，值越大分割越粗块越少
"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# 加载文本
with open("asset/load/09-ai1.txt", encoding="utf-8") as f:
	state_of_the_union = f.read()  # 返回字符串

# 获取嵌入模型
embed_model = OpenAIEmbeddings(
	model="text-embedding-3-large"
)
# 获取切割器
text_splitter = SemanticChunker(
	embeddings=embed_model,
	breakpoint_threshold_type="percentile",  # 断点阈值类型：字面值["百分位数", "标准差", "四分位距", "梯度"] 选其一
	breakpoint_threshold_amount=65.0  # 断点阈值数量 (极低阈值 → 高分割敏感度，块越多)。 计算相邻文本向量的余弦夹角，当夹角大于此值时，会切到不同的chunk中
)
# 切分文档
docs = text_splitter.create_documents(texts=[state_of_the_union])
print(len(docs))
for doc in docs:
	print(f"🐦 文档 {doc}")
