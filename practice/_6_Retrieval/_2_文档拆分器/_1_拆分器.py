"""
	1.  为什么要拆分？ 若将一整个Document使用时，存在的问题：
			a. 若用户提问的Query 出现在某个Document中，当将检索到的整个Document对象直接放入Prompt 中并不是最优的选择，因为若Query只出现在Document的某一个片段中，其他片段内容都是不相关的，那么这些无效信息
				越多，对LLM后续的推理影响越大；
			b. 任何一个大模型都存在最大token限制

	2. chunking 拆分策略：
			a.  根据句子拆分：  按照自然句子边界进行切分，以保证语义的完整性；

			b. 按照固定字符数切分：  可能会在不适当的位置切断句子

			c. 按照固定字符数切分，结合重叠窗口：  abcdefg -> 切分为  abcd  cdef ，后一段部分内容与前一段重叠。通过重叠窗口来避免切分关键内容

			d. 递归字符切分：
					通过指定某些字符动态确定切分点，可根据文档的复杂性和内容密度来调整块的大小。
					如： 以 \n 换行符切分， 切分后若块比较大 -> 继续以 。句号进行切分  -> 若块仍比较大 继续以固定字符数切分....

			e. 根据语义内容切分： 适用于需要高度语义保持的场景。 切分的块有可能大小不一，如： 80%内容语义相同  20%语义不同，那么切出的两块大小差别很大

	3. 各种文档拆分器的父类 - TextSplitter
			属性：
					chunk_size : 块的最大尺寸，默认 4000字符数。 由长度函数 length_function 进行字符数测量

					length_function：  用于测量给定块字符数的函数。默认为 len函数，len函数按unicode字符数计算，所以一个汉字、一个英文字母、一个符号都算一个字符。

					chunk_overlap:  相邻两个块之间的字符重叠数，默认200，通常设置为chunk_size 的 10% ～ 20%

					keep_separator:  是否在块中保留分隔符。默认False，如 以。句号分割块，默认切出来的块中就不包含句号了

					add_start_index:  是否在元数据中包含块的起始索引，默认False

					strip_whitespace:  是否从每个文档的开始和结束处 去除空白字符，默认True

					add_start_index:  True 时，切块后的每个Document中，metadata显示当前块的首字符在原文本中的位置

			常用方法：
				split_text(xx) :   传入 str ， 返回 list[str]

				create_document(xx):  传入 list[str] , 返回 list[Document] 。 将传入的字符串列表进行循环，将每个str元素调用 split_text，最后组装成Document返回

				split_documents(xx):   传入 list[Document]  , 返回 list[Document] 。 循环取出列表中Document的page_content，构成list[str]，调用 create_document()

				from_huggingface_tokenizer:  利用 huggingface 提供的tokenizer 来计算长度


	4.  了解的一些拆分器：官方文档中有提供各个类型的拆分器，使用时可以对应的找一下
			HTMLHeaderTextSplitter  : 处理HTML文档，根据标题标签将文档划分块，同时保留标题的层级结构

			CodeTextSplitter：代码文件的分割器，根据编程语言的语法结构拆分。与RecursiveCharacterTextSplitter 不同，CodeTextSplitter 针对代码进行了优化，能够避免在函数或类的中间截断。

			MarkdownTextSplitter：
"""