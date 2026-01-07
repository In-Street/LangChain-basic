import dotenv
from langchain_core.prompts import load_prompt

dotenv.load_dotenv()

prompt_template = load_prompt('prompt.yml', encoding='utf-8')
prompt_template_1 = load_prompt('prompt.json', encoding='utf-8')

prompt_value = prompt_template_1.invoke({'name': 'Jay', 'category': '晴天'})
print(prompt_value)
