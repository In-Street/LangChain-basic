"""
	 Chain:  用于将多个组件（提示词模版、大模型、输出解析器、记忆、工具）连接起来，形成复用的工作流，完成复杂的任务
	    1. 核心： 组合不同模块化单元
	            LLM 与 PromptTemplate 结合、LLM 与 输出解析器 结合、LLM 与 外部数据 结合、LLM 与 长期记忆 结合

		2. LCEL ： LangChain Expression Language，通过管道符 |  将组建连接成可执行流程， Prompt | model | OutputParse
				所有LCEL 组件对象都实现了 Runnable 协议， 强制所有组件实现一组标准方法：invoke 、batch、stream  。
				保证了一致的调用方式，如 PromptTemplate # invoke 、 ChatOpenAI # invoke 、 JsonOutputParse # invoke

		3. 数学链 - LLMMathChain.from_llm( model ) # invoke( ''10**3+100的结果是多少 )
				将用户问题转换为数学问题，将数学问题转换为用 numexpr 库执行的表达式。使用运行此代码的输出来回答问题
				pip3 install numexpr


		4. 路由链 - RouterChain
				用于创建可以动态选择下一条链的链。 可自动分析用户的需求，然后引导到最合适的链中执行，获取响应并返回最终结果。
				如：
				有三类chain，分别对应三种学科的问题解答。我们的输入内容也是与这三种学科对应， 但是随机的，比如第一次输入数学问题、第二次有可能是历史问题...
				这时候期待的效果是:可以根据输 入的内容是什么，自动将其应用到对应的子链中。RouterChain就为我们提供了这样一种能力。

		4. 文档链 - StuffDocumentsChain
				将多个文档内容合并到单个prompt中，然后传递到LLM处理。
				由于所有文档被完整拼接，LLM能同时看到全部内容，适合全局理解的任务，如：总结、问答、对比分析。适合处理 少量/中等长度文档
"""