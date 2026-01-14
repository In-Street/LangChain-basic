"""
	 1. 如何设计Memory:
	        1. 最直接的方式： 保留一个聊天消息列表。包括 HumanMessage、AIMessage
	                手动实现：
	                    template = ChatPromptTemplate.from_message(xxxx)
	                    在while True 循环中，
	                        将 LLM # invoke 的返回值response，再次塞到template中，template.messages.append( AIMessage(response.content) )
	                        将用户提问的内容，再次塞到template中，template.messages.append( HumanMessage(用户提问内容) )

	        2. 只返回最近交互的k条消息
	        3. 返回过去k条消息的摘要
	        4.  从存储的消息中提取实体，仅将实体的信息再次抛给LLM

	2. LangChain 中直接使用Memory的工具：

			1. 对话类：ConversationBufferMemory、 ConversationBufferWindowMemory

			2. 实体类：ConversationEntityMemory、InMemoryEntityStore、RedisEntityStore、SQLiteEntityStore

				ConversationEntityMemory：
					a. 基于实体的对话记忆机制，能够智能的识别、存储、利用对话中出现的实体信息（如 人名、地点）及其属性关系，并结构化存储，使AI具备更强的上下文理解和记忆能力
					b. 解决信息过载的问题：因为长对话中有大量冗余信息干扰，通过实体摘要，可压缩非重要的细节（如删除寒暄等，保留时间/价格等事实）
					c. 场景： 在医疗领域，必须用实体记忆确保关键信息，如过敏史，能被100%准确识别和拦截。 若使用ConversationSummaryMemory 时，LLM在生成摘要时可能会忽略过敏信息

			3. 摘要类：ConversationSummaryMemory 、ConversationSummaryBufferMemory
"""