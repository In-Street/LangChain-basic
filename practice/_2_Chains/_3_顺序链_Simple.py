"""
	顺序链： 允许将多个链顺序连接，每个Chain 的输出作为下一个Chain的输入，形成流水线.

	类型：
		单个输入/输出：SimpleSequentialChain  /sɪˈkwenʃl//

				input ->  LLMChain -> LLMChain -> output
				多个链串联执行，每步都有单一的输入和输出，上一步的输出就是下一步的输入，无需手动映射。 invoke 参数变量字典的key值 唯一固定为 input
				SimpleSequentialChain( chains=[LLMChain, LLMChain] )   # invoke( {'input': xxxx } )


		多个输入/输出： SequentialChain
"""
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain_core.prompts import ChatPromptTemplate
from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=300)

pt_1 = ChatPromptTemplate.from_messages(
	[
		('system', '你是一个资深音乐爱好者'),
		('human', '请详细列出：{content}')
	]
)

llm_chain_1 = LLMChain(llm=model, prompt=pt_1 )


pt_2 = ChatPromptTemplate.from_messages(
	[
		('system', '你是一个对音乐类型非常了解的资深者'),
		('human', '这是一个完整的歌手及其作品的内容：{content_2}'),
		('human','根据上述内容，简短的总结出歌手、作品、类型，控制在50字内')
	]
)

llm_chain_2 = LLMChain(llm=model, prompt=pt_2 )



# 定义完整的链，串联两个顺序执行的链
full_chain = SimpleSequentialChain(
	chains=[llm_chain_1, llm_chain_2]
)

# 对 SimpleSequentialChain 而言，唯一的输入变量名为 input
response = full_chain.invoke(input={'input': '2000～2010期间华语乐坛代表性歌手及其作品'})
print(type(response)) # dict
print(response) #  {'input': '2000～2010期间华语乐坛代表性歌手及其作品', 'output': '2000-2010年华语乐坛涌现周杰伦、陶喆、林俊杰等歌手，作品涵盖R&B、中国风、独立民谣等，推动流行音乐多元化发展。'}

