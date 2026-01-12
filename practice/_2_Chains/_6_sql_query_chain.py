"""
		pip3 install pymysql

		SQL查询链：用于将 自然语言 转换为 数据库SQL查询语句
"""
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

from practice.common_func import get_chat_model

chat_model = get_chat_model()

# 连接数据库
db = SQLDatabase.from_uri("mysql+pymysql://root:taylor18@localhost:3306/fastapi")
print(f'数据库类型:{db.dialect}')
print(f'可用数据表：{db.get_usable_table_names()}')
# db.run('select count(id) from ')  # 执行查询语句

# 创建 chain实例
chain = create_sql_query_chain(
	llm=chat_model,
	db=db
)

res = chain.invoke(input=
{
	'question': '数据表book中，以author分组，得出每组中price最高的数据',
	'table_names_to_use': ['book']
}
)
print(res)
# 输出：
# SELECT `author`, MAX(`price`) AS max_price
# FROM book
# GROUP BY `author`
# LIMIT 5;
