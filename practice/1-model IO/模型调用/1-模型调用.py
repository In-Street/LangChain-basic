import os, dotenv
from langchain_community.chat_models import ChatZhipuAI,zhipuai
from langchain_openai import ChatOpenAI,OpenAI


dotenv.load_dotenv()  # 加载 .env配置文件，从配置文件中读取 api_key 等信息

"""
    1. 非对话模型：
            输入：文本字符串 或 PromptValue 对象
            输出：返回文本字符串
            场景：单次文本生成任务（如摘要生成、翻译）
            不支持多轮对话上下文：每次调用独立处理，无法自动关联历史对话
    
    2. 对话模型：
            输入： list[Message] 或 PromptValue，每条消息需指定角色，如：SystemMessage、HumanMessage、AIMessage 。也支持字符串输入
            输出：返回带角色的消息对象（BaseMessage 子类），通常是AIMessage
            场景：对话系统（客服机器人、交互助手）
            原生支持多轮对话：通过消息列表维护上下文，模型可基于完整对话历史生成回复            
             
    1. temperature: 控制文本生成的随机性，取值 0～1
            值越低：输出越稳定、保守。适合事实回答
            值越高： 输出多样、有创意。适合创意写作
            
            精确模式：0.5或更低，生成的文本更加可靠，但缺乏创意和多样性
            平衡模式：0.8，生成的文本有一定的多样性，又保持较好的连贯性和准确性
            创意模式：1，但更易出现语法错误或不合逻辑的内容
        
    2. max_tokens: 限制生成文本的最大长度        
"""
chat_model = ChatZhipuAI(model='glm-4-flash-250414',
                         api_key=os.getenv('ZHIPUAI_API_KEY'),  # 从.env配置文件读取
                         api_base=os.getenv('ZHIPUAI_API_BASE') + '/chat/completions',
                         max_tokens=100,
                         temperature=0.8)

# 模型调用
# response = chat_model.invoke('学习大模型，首次调用，制定langchain的学习计划')
# print(response.content)


# ChatOpenAI 的属性api_key、base_url 若在创建实例时未传，默认情况下会从环境变量中获取。此处将配置文件中的值设置到环境变量中
os.environ['OPENAI_API_KEY'] = os.getenv('ZHIPUAI_API_KEY')
os.environ['OPENAI_BASE_URL'] = os.getenv('ZHIPUAI_API_BASE')

#使用OpenAI 来创建LLM对话模型实例，调用智谱
open_ai_chat_model = ChatOpenAI(model='glm-4-flash-250414',
                                # api_key=os.getenv('ZHIPUAI_API_KEY'),  # 未设定时，会从环境变量中获取
                                # base_url=os.getenv('ZHIPUAI_API_BASE'),
                                max_tokens=300, temperature=0.8)

# 返回结果类型： <class 'langchain_core.message.ai.AIMessage'>
invoke_response = open_ai_chat_model.invoke('制定一份langchain的学习计划')
print(invoke_response.content)
