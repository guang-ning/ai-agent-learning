# 嵌入式工程师 → AI Agent 工程师 转行闭环方案

## 📋 核心理念
**"做中学"** - 从第一天就写代码，在真实项目中补基础，3-4 个月内完成从零到可就业的转变。

---

## 🎯 12 周完整执行方案

### **第一周：快速破冰（5 天）**

**目标**：跑通第一个 Agent，建立信心

#### Day 1-2：环境搭建 + 第一个 API 调用

**第一个脚本** - `Week1/day1_hello_agent.py`：
```python
import os
from openai import OpenAI

# 设置 API 密钥
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 第一个对话
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "你好，我是一个嵌入式工程师，想转行 AI Agent，给我一个学习建议"}
    ]
)

print(response.choices[0].message.content)
```

**任务**：
- [ ] 申请 OpenAI API（或用国内替代：阿里通义、百度文心等）
- [ ] 创建 `.env` 文件存储 API Key
- [ ] 跑通这个脚本

**参考项目**：看这个项目的 README：[LangChain 官方示例](https://github.com/langchain-ai/langchain)

---

#### Day 3-4：理解 Agent 的三个核心概念

**边读边写代码**，不要只看理论：

1. **什么是 Prompt（提示词）**
   - 打开：[Hello-Agents 的 Prompt 章节](https://github.com/datawhalechina/hello-agents)
   - 动手：修改上面的脚本，加入 System Prompt
   ```python
   messages=[
       {"role": "system", "content": "你是一个 Python 代码审查专家"},
       {"role": "user", "content": "这段代码有问题吗？..."}
   ]
   ```
   - [ ] 试 5 种不同的 prompt，观察效果差异

2. **什么是 Tool（工具）**
   - 看这个例子：[LangChain Tool 使用](https://python.langchain.com/docs/modules/tools/)
   - 动手写个简单的 Tool：
   ```python
   def get_weather(city):
       """获取天气"""
       return f"{city} 今天晴天，25°C"
   
   # Agent 需要时可以调用这个函数
   ```

3. **什么是 Memory（记忆）**
   - 对话历史的管理
   - 动手：保存对话历史到列表，实现多轮对话

**参考开发��代码**：
- [LangChain 官方示例库](https://github.com/langchain-ai/langchain/tree/master/templates)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

---

#### Day 5：完成第一个迷你项目

**项目名**：`SimpleQAAgent`

需求：
- 创建一个能回答"关于我"问题的 Agent
- Agent 有 3 个工具：get_age()、get_job()、get_skills()
- 支持多轮对话

**代码框架** - `Week1/day5_qa_agent.py`：
```python
from openai import OpenAI
import json

class SimpleAgent:
    def __init__(self):
        self.client = OpenAI()
        self.tools = {
            "get_age": lambda: "28",
            "get_job": lambda: "嵌入式工程师",
            "get_skills": lambda: "C, Python, 嵌入式系统"
        }
        self.conversation = []
    
    def process_tool_call(self, tool_name):
        """执行工具调用"""
        if tool_name in self.tools:
            return self.tools[tool_name]()
        return "工具不存在"
    
    def chat(self, user_message):
        """多轮对话"""
        self.conversation.append({"role": "user", "content": user_message})
        
        # 调用 LLM
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.conversation
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message

# 测试
if __name__ == "__main__":
    agent = SimpleAgent()
    print(agent.chat("你今年多少岁？"))
    print(agent.chat("你的职位是什么？"))
```

**任务**：
- [ ] 完成代码并测试
- [ ] Push 到 GitHub
- [ ] 写一个简单的 README

---

### **第 2-3 周：学习 LangChain 框架（做中学）**

**目标**：掌握 Agent 开发的标准工具链

#### Week 2：LangChain 基础

参考项目：[LangChain Python 官方文档](https://python.langchain.com/)

**项目**：`CodeReviewAgent`

需求：
1. Agent 能读取代码文件
2. 调用 LLM 进行代码审查
3. 给出改进建议，并保存为 markdown 报告

代码框架 - `Week2/code_review_agent.py`：
```python
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain import hub

@tool
def read_code_file(file_path: str) -> str:
    """读取代码文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败: {e}"

@tool
def save_report(content: str, output_path: str) -> str:
    """保存代码审查报告"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"报告已保存到 {output_path}"
    except Exception as e:
        return f"保存失败: {e}"

def create_code_review_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [read_code_file, save_report]
    
    prompt = hub.pull("hwchase17/openai-functions-agent-prompt")
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    return agent_executor

if __name__ == "__main__":
    agent = create_code_review_agent()
    result = agent.invoke({
        "input": "请审查 my_script.py 文件中的代码质量，然后将报告保存到 review_report.md"
    })
    print(result)
```

**任务**：
- [ ] 跑通 LangChain 官方示例
- [ ] 定义 5 个自己的工具
- [ ] 完成代码审查 Agent

---

#### Week 3：RAG 系统（检索增强生成）

参考项目：[LangChain RAG 官方示例](https://python.langchain.com/docs/use_cases/question_answering/)

**项目**：`DocumentQAAgent`

需求：
1. 用户上传文档（PDF/TXT）
2. Agent 索引文档
3. 回答关于文档内容的问题
4. 支持多轮对话和上下文记忆

代码框架 - `Week3/document_qa_agent.py`：
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA

def create_rag_agent(document_path: str):
    # Step 1：加载文档
    loader = TextLoader(document_path)
    documents = loader.load()
    
    # Step 2：分割文本
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    # Step 3：生成向量 + 保存
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Step 4：创建 RAG Chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa

if __name__ == "__main__":
    qa_agent = create_rag_agent("my_document.txt")
    
    # 多轮对话
    questions = [
        "这份文档主要讲什么？",
        "有什么关键要点？",
        "如何应用到实际项目中？"
    ]
    
    for q in questions:
        result = qa_agent.run(q)
        print(f"Q: {q}")
        print(f"A: {result}\n")
```

**参考优秀开发者代码**：
- [Langchain-Chatchat（中文）](https://github.com/chatchat-space/Langchain-Chatchat)
- [PrivateGPT](https://github.com/imartinez/privateGPT)

---

### **第 4-6 周：多 Agent 系统设计**

**目标**：从单 Agent 升级到 Agent 编排

参考项目：
- [MetaGPT](https://github.com/geekan/MetaGPT) - 多 Agent 协作框架
- [AutoGen](https://github.com/microsoft/autogen) - Agent 通信框架

#### Week 4-6：**完整项目** - `AITeamAssistant`

需求：
- 模拟一个小团队（产品经理、开发、测试）
- 用户提需求
- 多个 Agent 协作完成任务
- 输出最终交付物

**核心架构** - `Week4-6/ai_team_assistant/`：
```
├── agents/
│   ├── product_manager.py
│   ├── developer.py
│   ├── tester.py
│   └── orchestrator.py
├── tools/
│   ├── code_generator.py
│   ├── test_runner.py
│   └── document_writer.py
├── main.py
└── README.md
```

代码框架 - `Week4-6/ai_team_assistant/orchestrator.py`：
```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent

class AITeamOrchestrator:
    def __init__(self):
        self.product_manager = self.create_pm_agent()
        self.developer = self.create_dev_agent()
        self.tester = self.create_test_agent()
    
    def create_pm_agent(self):
        """产品经理 Agent"""
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        # 定义 PM 的工具和 prompt
        return llm
    
    def create_dev_agent(self):
        """开发者 Agent"""
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        # 定义开发者的工具和 prompt
        return llm
    
    def create_test_agent(self):
        """测试者 Agent"""
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        # 定义测试者的工具和 prompt
        return llm
    
    def execute_workflow(self, user_requirement: str):
        """执行完整工作流"""
        # Step 1：PM 分解需求
        plan = self.product_manager(user_requirement)
        
        # Step 2：开发者编码
        code = self.developer(plan)
        
        # Step 3：测试者验证
        test_result = self.tester(code)
        
        return {
            "plan": plan,
            "code": code,
            "test_result": test_result
        }

if __name__ == "__main__":
    team = AITeamOrchestrator()
    result = team.execute_workflow("开发一个计算器应用")
    print(result)
```

**参考优秀项目**：
- [MetaGPT 官方示例](https://github.com/geekan/MetaGPT/tree/main/examples)
- [AutoGen 多 Agent 例子](https://github.com/microsoft/autogen/tree/main/notebook)

---

### **第 7-9 周：工程化 + 优化**

**目标**：让项目从"可跑"升级到"可用"

#### Week 7：**错误处理 + 日志 + 配置管理**

代码框架 - `Week7-9/best_practices/`：
```
├── config.yaml         # 配置管理
├── logger_setup.py     # 日志系统
├── error_handler.py    # 错误处理
├── rate_limiter.py     # 限流
└── monitoring.py       # 监控指标
```

**logger_setup.py**：
```python
import logging
import logging.handlers

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志系统"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger
```

**任务**：
- [ ] 为之前的项目加上日志系统
- [ ] 实现 API 调用重试机制
- [ ] 配置化管理 LLM 参数

---

#### Week 8：**性能优化 + 成本控制**

**参考**：[OpenAI 官方成本优化](https://platform.openai.com/docs/guides/tokens)

```python
from langchain.callbacks import OpenAICallbackHandler

handler = OpenAICallbackHandler()
# 追踪 Token 使用和成本
```

**任务**：
- [ ] 实现 Token 计数和成本估算
- [ ] 添加缓存机制（避免重复调用）
- [ ] 测试不同模型的性能差异

---

#### Week 9：**部署 + API 服务化**

将 Agent 包装成 API 服务 - `Week7-9/deployment/main_api.py`：

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    # 在这里调用你的 Agent
    result = agent.invoke({"input": query.question})
    return {"answer": result}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Dockerfile**：
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0"]
```

**任务**：
- [ ] 将某个 Agent 项目打包成 REST API
- [ ] 用 Docker 容器化
- [ ] 部署到云平台（AWS/阿里云/Vercel）

---

### **第 10-12 周：总结 + 开源 + 面试准备**

#### Week 10：**项目整理 + 开源贡献**

**GitHub 作品集要求**：
- 3-4 个完整项目
- 每个项目都有详细 README（包括原理 + 使用说明）
- 代码规范、有测试、有日志

**开源贡献**：
- 给 Dify/LangChain/AutoGen 提交 1-2 个 PR
- 或修复 bug，或添加文档示例

#### Week 11：**写技术博客 + 总结**

**博客方向**：
1. "从嵌入式到 AI Agent 的技能迁移"
2. "我用 LangChain 构建的 3 个项目"
3. "Agent 系统设计的最佳实践"

**发布平台**：
- Medium、Dev.to（英文）
- 掘金、知乎、CSDN（中文）

#### Week 12：**面试准备**

**面试高频问题**：
1. 你做过哪些 Agent 项目？
2. 如何设计一个多 Agent 协作系统？
3. RAG 系统的架构是什么？
4. 如何评估 Agent 的质量？
5. Token 成本如何优化？

---

## 📁 最终项目结构

```
ai-agent-learning/
├── Week1/
│   ├── day1_hello_agent.py
│   ├── day5_qa_agent.py
│   ├── requirements.txt
│   └── README.md
├── Week2/
│   ├── code_review_agent.py
│   ├── requirements.txt
│   └── README.md
├── Week3/
│   ├── document_qa_agent.py
│   ├── test_documents/
│   ├── requirements.txt
│   └── README.md
├── Week4-6/
│   ├── ai_team_assistant/
│   │   ├── agents/
│   │   ├── tools/
│   │   ├── main.py
│   │   └── README.md
│   └── requirements.txt
├── Week7-9/
│   ├── best_practices/
│   ├── deployment/
│   │   ├── main_api.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── monitoring/
├── .env.example
├── .gitignore
└── README.md（总体项目说明）
```

---

## 🎓 学习资源导航

| 阶段 | 资源 | 推荐度 |
|------|------|------|
| **Week 1-2** | [LangChain 官方文档](https://python.langchain.com/) | ⭐⭐⭐⭐⭐ |
| | [Hello-Agents](https://github.com/datawhalechina/hello-agents) | ⭐⭐⭐⭐ |
| **Week 3** | [LangChain RAG 指南](https://python.langchain.com/docs/use_cases/question_answering/) | ⭐⭐⭐⭐⭐ |
| | [Langchain-Chatchat（中文 RAG 参考）](https://github.com/chatchat-space/Langchain-Chatchat) | ⭐⭐⭐⭐ |
| **Week 4-6** | [AutoGen](https://github.com/microsoft/autogen) | ⭐⭐⭐⭐⭐ |
| | [MetaGPT](https://github.com/geekan/MetaGPT) | ⭐⭐⭐⭐⭐ |
| **Week 7-9** | [LangChain 生产最佳实践](https://python.langchain.com/docs/guides/production) | ⭐⭐⭐⭐ |
| | [Dify 源代码](https://github.com/langgenius/dify) | ⭐⭐⭐⭐ |
| **Week 10-12** | [AI-AGENT-Content-Hub](https://github.com/potatoImp/AI-AGENT-Content-Hub) | ⭐⭐⭐⭐ |

---

## ⚡ 快速启动

**第一天做这些**：

```bash
# 1. Clone 仓库
git clone https://github.com/guang-ning/ai-agent-learning.git
cd ai-agent-learning

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 3. 安装基础包
pip install -r Week1/requirements.txt

# 4. 创建 .env
cp .env.example .env
# 编辑 .env，添加你的 API Key

# 5. 运行第一个脚本
python Week1/day1_hello_agent.py

# 6. Push 你的修改
git add .
git commit -m "Day 1: First Agent"
git push
```

**每周节奏**：
- 📖 周一：学习理论（看文档 1-2 小时）
- 💻 周二-四：写代码 + 测试（每天 2-3 小时）
- 📤 周五：提交项目 + 写周总结

---

## 🎯 成功指标

| 阶段 | 完成标准 |
|------|--------|
| Week 1-2 | 能用 LangChain 写一个多轮对话的 Agent |
| Week 3 | 能构建本地 RAG 系统，回答文档问题 |
| Week 4-6 | 能设计多 Agent 协作流程，自动编排任务 |
| Week 7-9 | 能把 Agent 部署成 API 服务，可用于生产 |
| Week 10-12 | GitHub 有 3-4 个完整项目，能参加面试 |

---

## 💡 关键建议

1. **不要完美主义** → 先跑通，再优化
2. **每周必须写代码** → 看再多资料也白搭
3. **多看优秀项目源码** → 学习工程实践
4. **记录遇到的问题** → 做好笔记，面试时用
5. **定期复盘** → 每周反思学了什么、哪里还不懂

---

## 📞 遇到问题怎么办？

- **环境问题** → 看官方文档 + StackOverflow
- **代码 BUG** → 查看 GitHub Issues
- **API 限制** → 用国内替代（通义、文心等）
- **模型成本高** → 用 gpt-3.5-turbo 测试，再用 gpt-4
- **不会设计架构** → 参考 MetaGPT/AutoGen 的源码
