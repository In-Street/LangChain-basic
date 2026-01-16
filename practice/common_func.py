import os, dotenv
from typing import Optional
from langchain_openai import ChatOpenAI
from openai import OpenAI
import tiktoken

dotenv.load_dotenv()


def get_chat_model(temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                   streaming: Optional[bool] = False) -> ChatOpenAI:
	# OpenAI()
	return ChatOpenAI(
		model=os.getenv('GUIJI_MODEL_NAME'),
		api_key=os.getenv('GUIJI_API_KEY'),
		base_url=os.getenv('GUIJI_API_BASE'),
		temperature=temperature,
		streaming=streaming,
		max_tokens=max_tokens,
		# 使用Qwen3-8B 模型时，指定max_tokens无效，因为并没有实现OpenAI计算多轮对话token数的get_num_tokens_from_messages() 方法
	)

def get_model(temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                   streaming: Optional[bool] = False) -> OpenAI:

	return OpenAI(
		api_key=os.getenv('GUIJI_API_KEY'),
		base_url=os.getenv('GUIJI_API_BASE'),
	)


# 初始化Qwen3-8B对应的tiktoken编码器（固定用cl100k_base，无误差）
encoding = tiktoken.get_encoding('cl100k_base')

# Qwen的特殊控制token，固定值
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# 定义Qwen3-8B的多轮对话token计算核心函数，完全对标OpenAI的同名方法
def get_token_num_for_qwen(messages, encoding):
	"""
	    计算Qwen3-8B多轮对话messages的token数，完美替代原生缺失的方法
	    :param messages: 标准格式的对话列表，如 [{"role": "system", "content": "xxx"}, {"role": "user", "content": "xxx"}]
	    :param encoding: tiktoken编码器实例
	    :return: 总token数（含所有特殊控制token）
	    """
	num_tokens = 0
	for msg in messages:
		# 每条消息的固定格式：<|im_start|>role\ncontent<|im_end|>
		num_tokens += 4  # 基础固定token：<|im_start|>(1) + role换行符(1) + <|im_end|>(1) + 换行分隔(1)
		num_tokens += len(encoding.encode(msg["role"]))  # role的token数
		num_tokens += len(encoding.encode(msg["content"]))  # content的token数
	# 最后额外加1个token（Qwen生成回复时的结束占位，不影响计算精度，可加可不加）
	num_tokens += 2
	return num_tokens


# 测试：标准的messages多轮对话格式
messages = [
    {"role": "system", "content": "你是一个专业的编程助手，回答简洁准确"},
    {"role": "user", "content": "Qwen3-8B没有get_num_tokens_from_messages，有什么替代方案？"},
    {"role": "assistant", "content": "有两个最优方案，分别是基于tiktoken的高效实现和基于transformers的原生实现，都可以完美替代。"}
]

# 计算token数
# token_count = get_token_num_for_qwen(messages, encoding)
# print(f"messages总token数：{token_count}")