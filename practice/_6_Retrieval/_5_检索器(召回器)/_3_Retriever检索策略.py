"""

"""
from _2_Retriever的检索_数据准备 import db_for_retriever

"""
	Retriever 默认使用相似性检索
"""
retriever = db_for_retriever.as_retriever(
	search_kwargs={"k": 3}  # 返回文档数为3
)
res = retriever.invoke("经济政策")

for i, doc in enumerate(res):
	print(f'结果 {i + 1}：{doc}')
#输出：返回3条文档
# 结果 1：page_content='经济复苏：美国经济正在从疫情中强劲复苏，失业率降至历史低点。！'
# 结果 2：page_content='外交政策：继续支持乌克兰对抗俄罗斯的侵略。'
# 结果 3：page_content='移民改革：提出全面的移民改革方案。'

"""
	分数阈值查询： 只有相似度超过该值的文档才会返回
		只会返回满足阈值分数的文档，不会获取文档的得分。【文档得分可使用向量数据库提供的 similarity_search_with_relevance_scores 查看】
		
		similarity_score_threshold 、 score_threshold 配套使用，指定查询类型及分数阈值
"""
print("=" * 50)
retriever_2 = db_for_retriever.as_retriever(
	search_type="similarity_score_threshold",
	search_kwargs={
		"score_threshold": 0.3  # 分值大于0.3 的才会返回
	}
)
res_2 = retriever_2.invoke("经济政策")

for i, doc in enumerate(res_2):
	print(f'结果 {i + 1}：{doc}')
#输出：只有一条文档了
# 结果 1：page_content='经济复苏：美国经济正在从疫情中强劲复苏，失业率降至历史低点。！'

# scores = db_for_retriever.similarity_search_with_relevance_scores("经济政策")
# for i,(doc,score) in enumerate(scores):
# 	print(f'{doc}，分值：{score}')

"""
	MMR 搜索
"""

retriever_3 = db_for_retriever.as_retriever(
	search_type="mmr",
	search_kwargs={
		"fetch_k" : 2
	}
)
