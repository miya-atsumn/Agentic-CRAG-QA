import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 1. 初始化 DeepSeek 客户端（兼容 OpenAI SDK）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 2. 定义本地工具函数（示例：一个计算器）
def calculator(expression: str) -> str:
    """一个简单的计算器工具，接受数学表达式字符串。"""
    try:
        # 注意：生产环境应使用更安全的求值方式，这里仅为演示
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 3. 工具注册表：名称 -> 函数
TOOL_FUNCTIONS = {
    "calculator": calculator,
}

# 4. 工具声明（告诉 LLM 有哪些工具可用，格式遵循 JSON Schema）
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "一个计算器，可以计算数学表达式，例如 '3 * (5 + 2)'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# 5. 核心 ReAct 循环
def run_agent(user_query, max_steps=5):
    messages = [{"role": "user", "content": user_query}]
    
    for step in range(max_steps):
        print(f"\n--- 第 {step + 1} 步 ---")
        
        # 调用 LLM，传入消息历史和工具列表
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让模型自行决定是否调用工具
        )
        
        msg = response.choices[0].message
        
        # 情况 A：模型给出了最终答案，退出循环
        if not msg.tool_calls:
            print(f"模型最终回答: {msg.content}")
            return msg.content
        
        # 情况 B：模型请求调用工具
        # 首先，将模型的助手消息（包含 tool_calls）追加到历史中
        messages.append(msg)
        
        # 遍历所有工具调用请求（模型可能一次性请求多个）
        for tool_call in msg.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"模型请求调用工具: {function_name}，参数: {function_args}")
            
            # 从注册表中查找并执行函数
            if function_name in TOOL_FUNCTIONS:
                try:
                    observation = TOOL_FUNCTIONS[function_name](**function_args)
                except Exception as e:
                    # 异常回注：工具失败时将错误信息作为观察结果返回给模型
                    observation = f"工具执行异常: {str(e)}"
            else:
                observation = f"错误: 未知的工具 '{function_name}'"
            
            print(f"工具执行结果: {observation}")
            
            # 将工具执行结果作为 "tool" 角色的消息追加到历史中
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation
            })
        
        # 循环继续，带着工具结果再次调用 LLM
    print("达到最大步数限制，未能得出最终答案。")
    return None

# 6. 运行测试
if __name__ == "__main__":
    # 测试一个需要调用工具的问题
    run_agent("请帮我计算 11 乘以 22 等于多少？")