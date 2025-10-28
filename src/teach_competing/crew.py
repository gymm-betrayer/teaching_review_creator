import os
import uuid
import json
from pydantic import BaseModel
import re
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv
import pdfplumber
from pathlib import Path
from openai import OpenAI
from uuid import uuid4
from typing import List, Dict, Any, Optional
from crewai.tools import BaseTool
import chromadb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

load_dotenv()

dashscope_client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                          base_url=os.getenv("DASHSCOPE_URL"))

def extract_text_from_pdf(pdf_path: str) -> str:
    """从单个 PDF 文件提取纯文本"""
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"⚠️ 无法读取 {pdf_path}: {e}")
        return ""

def load_all_pdfs_from_directory(directory: str) -> List[Dict[str, str]]:
    """
    加载目录下所有 PDF，返回 [{'filename': ..., 'text': ...}, ...]
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"目录 {directory} 中没有找到 .pdf 文件")

    documents = []
    for pdf_file in pdf_files:
        print(f"📄 正在读取: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))
        if text.strip():
            documents.append({
                "filename": pdf_file.stem,  # 不带 .pdf 后缀
                "text": text
            })
        else:
            print(f"⚠️ 跳过空文件: {pdf_file.name}")

    return documents

def split_pdf_into_chapters(text: str) -> List[Dict[str, str]]:
    """
    将学术 PDF 文本按章节切分，排除参考文献及之后内容
    返回章节列表，每个章节含：title, content, level (1/2/3)
    """
    # Step 1: 截断“参考文献”及之后内容
    ref_match = re.search(r'^\s*参考文献\s*$', text, re.MULTILINE | re.IGNORECASE)
    if ref_match:
        text = text[:ref_match.start()]

    # Step 2: 按行分割
    lines = text.split('\n')

    # Step 3: 定义标题正则（按优先级）
    patterns = [
        (1, re.compile(r'^\s*([一二三四五六七八九十]+、)\s*(.+)$')),  # 一、XXX
        (2, re.compile(r'^\s*（([一二三四五六七八九十]+)）\s*(.+)$')),  # （一）XXX
        (3, re.compile(r'^\s*(\d+)\.\s+(.+)$')),  # 1. XXX
    ]

    chapters = []
    current = None

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # 检查是否为标题
        is_title = False
        for level, pattern in patterns:
            match = pattern.match(line)
            if match:
                # 保存上一章
                if current:
                    chapters.append(current)

                # 新建章节
                if level == 1:
                    title = match.group(0).strip()
                elif level == 2:
                    title = f"（{match.group(1)}）{match.group(2)}"
                else:  # level == 3
                    title = f"{match.group(1)}. {match.group(2)}"

                current = {
                    "title": title,
                    "content": "",
                    "level": level
                }
                is_title = True
                break

        if not is_title:
            if current is None:
                # 开头段落（摘要后、第一章前）→ 归入虚拟“引言”
                if not chapters and any(kw in line for kw in ["ChatGPT", "人工智能时代", "本文将"]):
                    current = {
                        "title": "引言",
                        "content": line + "\n",
                        "level": 1
                    }
                else:
                    # 忽略封面、作者、摘要等（可根据需要调整）
                    continue
            else:
                current["content"] += line + "\n"

    # 添加最后一章
    if current:
        chapters.append(current)

    # 合并“引言”到第一章（可选）
    if len(chapters) >= 2 and chapters[0]["title"] == "引言":
        intro = chapters.pop(0)
        chapters[0]["content"] = intro["content"] + chapters[0]["content"]
        chapters[0]["title"] = f"{chapters[0]['title']}（含引言）"

    return chapters


def ingest_pdfs_to_chromadb(
    pdf_dir: str,
    collection_name: str,
    persist_directory: str = "./chroma_db",
    model_name: str = "BAAI/bge-small-zh-v1.5"
) -> chromadb.Collection:
    """
    从指定目录读取所有 PDF，自动切分章节，并存入 ChromaDB。
    如果集合已存在，则加载并复用（需重新提供 embedding function）。
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF 目录不存在: {pdf_dir}")

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"目录 {pdf_dir} 中没有 .pdf 文件")

    print(f"📁 找到 {len(pdf_files)} 个 PDF 文件，开始处理...")

    # 初始化 ChromaDB 客户端
    client = PersistentClient(path=persist_directory)

    # 初始化 embedding function（必须在 create 和 get 时都使用）
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device="cpu"
    )

    # 尝试获取现有集合；如果不存在，则创建
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func  # ← 关键：必须传！
        )
        print(f"📂 加载现有集合: {collection_name}")
    except NotFoundError:
        print(f"🆕 集合 {collection_name} 不存在，正在创建...")
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

    # 批量处理 PDF
    all_documents = []
    all_metadatas = []
    all_ids = []
    global_idx = 0

    for pdf_file in pdf_files:
        print(f"📄 处理: {pdf_file.name}")
        raw_text = extract_text_from_pdf(str(pdf_file))
        if not raw_text.strip():
            print(f"   ⚠️ 跳过空文件")
            continue

        chapters = split_pdf_into_chapters(raw_text)
        source_name = pdf_file.stem  # 不带 .pdf 后缀

        for chap in chapters:
            content = chap["content"].strip()
            if len(content) < 50:  # 过滤太短的片段
                continue

            doc_id = f"{collection_name}_{global_idx:06d}"
            all_documents.append(content)
            all_metadatas.append({
                "title": chap["title"],
                "level": chap["level"],
                "source": source_name,
                "chunk_type": "chapter" if chap["level"] == 1 else "subsection",
                "index": global_idx
            })
            all_ids.append(doc_id)
            global_idx += 1

    if not all_documents:
        raise ValueError("未提取到任何有效章节内容")

    # 批量添加到集合
    collection.add(
        documents=all_documents,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print(f"✅ 成功入库 {len(all_documents)} 个章节（来自 {len(pdf_files)} 个 PDF）")
    print(f"🧠 使用模型: {model_name}")
    print(f"💾 存储路径: {persist_directory}")
    return collection


def retrieve_from_chromadb(
        query: str,
        collection_name: str,
        n_results: int = 3,
        where_filter: Optional[Dict[str, Any]] = None,
        persist_directory: str = "./chroma_db"
) -> List[Dict[str, Any]]:
    """
    从 ChromaDB 检索相关章节，并返回内容 + level 等元数据

    Args:
        query: 用户查询文本
        collection_name: 集合名称
        n_results: 返回结果数量
        where_filter: ChromaDB 过滤条件，如 {"level": 1}
        persist_directory: 数据库存储路径

    Returns:
        List of dict with keys: content, level, title, source, score
    """
    # 初始化客户端
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(collection_name)

    # 执行查询（自动 embedding）
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # 解析结果
    retrieved = []
    for i in range(len(results["ids"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # 转换 distance 为 similarity score（Chroma 默认用 L2 或 cosine 距离）
        # 如果使用 cosine 距离：similarity = 1 - distance
        # 如果使用内积（IP）：需特殊处理，但我们用的是 cosine（bge 默认）
        score = 1.0 - distance

        retrieved.append({
            "content": doc,
            "level": meta["level"],  # int: 1, 2, 3...
            "title": meta["title"],  # str
            "source": meta["source"],  # str
            "chunk_type": meta["chunk_type"],  # str
            "score": round(score, 4)  # float, 0~1
        })

    # 按 score 降序（虽然 Chroma 已排序，但保险起见）
    retrieved.sort(key=lambda x: x["score"], reverse=True)

    return retrieved


class ChromaDBSearchTool(BaseTool):
    name: str = "ChromaDB 语义检索工具"
    description: str = (
        "根据自然语言查询，从本地 ChromaDB 向量数据库中检索相关学术章节内容。"
        "返回结果包含来源文件、章节标题、层级和具体内容，适用于回答教育、政策类问题。"
    )

    # 可配置参数（通过实例化时传入）
    collection_name: str = "ai_research"
    persist_directory: str = "./chroma_db"
    n_results: int = 4

    def _run(self, query: str, level_filter: Optional[int] = None) -> str:
        """
        执行检索

        Args:
            query (str): 用户的自然语言问题
            level_filter (int, optional): 仅检索指定层级（1=章，2=节，3=小节）

        Returns:
            str: 格式化的检索结果文本
        """
        # 构建 where_filter
        where_filter: Optional[Dict[str, Any]] = None
        if level_filter is not None:
            where_filter = {"level": level_filter}

        try:
            results: List[Dict[str, Any]] = retrieve_from_chromadb(
                query=query,
                collection_name=self.collection_name,
                n_results=self.n_results,
                where_filter=where_filter,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            return f"⚠️ 检索失败: {str(e)}"

        if not results:
            return "未在知识库中找到相关内容。"

        # 格式化为 LLM 友好文本
        parts = []
        for i, r in enumerate(results, 1):
            part = (
                f"【参考片段 {i}】\n"
                f"- 来源: {r['source']}\n"
                f"- 标题: {r['title']} (层级 {r['level']})\n"
                f"- 内容: {r['content'].strip()}\n"
            )
            parts.append(part)

        return "\n".join(parts)


@CrewBase
class TeachCompeting():
    """TeachCompeting crew"""
    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        self._collection = None
        self._search_tool = None

    qwen_max = LLM(
        model="dashscope/qwen-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.6,
    )

    qwen3_max =LLM(
        model="dashscope/qwen3-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.6,
    )
    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @property
    def collection(self):
        if self._collection is None:
            self._collection = ingest_pdfs_to_chromadb(
                pdf_dir="./arxiv_PDF",  # 相对于运行目录
                collection_name="ai_research"
            )
        return self._collection

    @property
    def search_tool(self):
        if self._search_tool is None:
            self._search_tool = ChromaDBSearchTool()
        return self._search_tool

    @agent
    def searcher(self) -> Agent:
        return Agent(
            config=self.agents_config['searcher'], # type: ignore[index]
            verbose=True,
            llm=self.qwen_max,
            tools=[self.search_tool]
        )

    @agent
    def idea_thinker(self) -> Agent:
        return Agent(
            config=self.agents_config['idea_thinker'],  # type: ignore[index]
            verbose=True,
            llm=self.qwen_max
        )

    @agent
    def writer1(self) -> Agent:
        return Agent(
            config=self.agents_config['writer1'], # type: ignore[index]
            verbose=True,
            llm=self.qwen_max,
        )

    @agent
    def writer2(self) -> Agent:
        return Agent(
            config=self.agents_config['writer2'],  # type: ignore[index]
            verbose=True,
            llm=self.qwen3_max,
        )

    @agent
    def polisher(self) -> Agent:
        return Agent(
            config=self.agents_config['polisher'],  # type: ignore[index]
            verbose=True,
            llm=self.qwen3_max,
        )
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def searcher_task(self) -> Task:
        return Task(
            config=self.tasks_config['searcher'], # type: ignore[index]
            agent=self.searcher()
        )

    @task
    def idea_thinker_task(self) -> Task:
        return Task(
            config=self.tasks_config['idea_thinker'],  # type: ignore[index]
            agent=self.idea_thinker(),
        )

    @task
    def writer1_task(self) -> Task:
        return Task(
            config=self.tasks_config['writer1'], # type: ignore[index]
            agent=self.writer1(),
            human_input=True
        )

    @task
    def writer2_task(self) -> Task:
        return Task(
            config=self.tasks_config['writer2'],  # type: ignore[index]
            output_file='article1.md',
            agent=self.writer2(),
            context=[self.writer1_task(), self.searcher_task(), self.idea_thinker_task()]
        )

    @task
    def polisher_task(self) -> Task:
        return Task(
            config=self.tasks_config['polisher'],  # type: ignore[index]
            output_file='article2.md',
            agent=self.polisher()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TeachCompeting crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
