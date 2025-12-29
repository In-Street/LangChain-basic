import os, dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage


dotenv.load_dotenv()  # 加载 .env配置文件

"""
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


#使用OpenAI 来创建LLM实例，调用智谱
open_ai_chat_model = ChatOpenAI(model='glm-4-flash-250414',
                                api_key=os.getenv('ZHIPUAI_API_KEY'),
                                base_url=os.getenv('ZHIPUAI_API_BASE'),
                                max_tokens=300, temperature=0.8)

invoke_response = open_ai_chat_model.invoke('制定一份langchain的学习计划')
print(invoke_response.content)
