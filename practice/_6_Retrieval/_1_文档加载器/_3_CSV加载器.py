"""

"""
from langchain_community.document_loaders import CSVLoader

csv_loader = CSVLoader(
	file_path="../../resources/asset/load/03-load.csv"
)

docs = csv_loader.load()

print(d for d in docs)
print([d for d in docs])  # 列表推导式
