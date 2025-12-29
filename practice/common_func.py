import os, dotenv
from typing import Optional
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


def get_chat_model(temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                   streaming: Optional[bool] = False) -> ChatOpenAI:
	return ChatOpenAI(
		model=os.getenv('ZHIPUAI_MODEL_NAME'),
		api_key=os.getenv('ZHIPUAI_API_KEY'),
		base_url=os.getenv('ZHIPUAI_URL'),
		temperature=temperature,
		max_tokens=max_tokens,
		streaming= streaming
	)
