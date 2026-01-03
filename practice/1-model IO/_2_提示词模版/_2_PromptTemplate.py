"""
	PromptTemplate:
		1. 获取实例：
				a. 使用构造方法，必传参数： template 、 input_variables

				b. 通过类方法 PromptTemplate.from_template(template=xxx)  -- 推荐使用

		2. 部分提示词模版的使用：
				a.  参数形式： from_template( partial_variables ={} ) , 指定部分变量的value。在format时再次指定会覆盖前面的value
				b.  实例方法形式：partial(key:value) 。 tem2 = tem1.partial(xx)，此方法不会对调用者tem1产生影响，而是返回一个新模版 tem2 产生影响

		3. 给变量赋值的两种方式：
				a. format() : 返回类型 str
				b. invoke(dict)：返回类型 PromptValue  -- 推荐使用

		4. 结合大模型使用：
				model.invoke(PromptValue)
"""
from langchain_core.prompts import PromptTemplate

# 1. 通过构造方法获取实例。
template = PromptTemplate(
	template='你是一个{role}，你的名字是{name}',
	input_variables=['role', 'name']
)

template_str = template.format(role='AI助手', name='小智')  # 变量赋值
print(template_str)

# 1.  通过 类方法： from_template 获取实例
template_2 = PromptTemplate.from_template(template='你是一个{role}，你的名字是{name}')
print(template_2.format(role='艺人', name='Jay'))



# 2. 部分提示词模版，通过参数 partial_variables 指定部分变量值
template_3 = PromptTemplate.from_template(
	template='你是一个{role}，你的名字是{name}，代表作:{collection}',
	partial_variables={'name': '周杰伦'})

print(  '部分提示词模版，参数形式：'+ template_3.format(role='创作者',collection = '晴天、枫'))


# 2. 部分提示词模版，通过实例方法 partial()  指定部分变量值
template_4 = PromptTemplate.from_template(
	template='你是一个{role}，你的名字是{name}，代表作:{collection}')

template_4_1 = template_4.partial(role='创作者_2', name='王力宏')
print(  '部分提示词模版，实例方法形式：'+ template_4_1.format(collection='爱错'))


#3.  通过 invoke() 方法给变量赋值
prompt_value = template_4_1.invoke(input={'collection': '大城小爱'})
print(f'通过invoke 方法给变量赋值: {prompt_value}')


# 4. 结合大模型使用
from practice.common_func import get_chat_model

model = get_chat_model(max_tokens=20)  # 获取模型
template_ = PromptTemplate.from_template(template='请评价{product}的优缺点，包括{param1}和{param2}')  # 生成提示词模版
prompt_value = template_.invoke(input={'product': 'Mac', 'param1': '处理器', 'param2': '系统优化'})  # 给模版变量赋值
response = model.invoke(prompt_value)  # 调用大模型
