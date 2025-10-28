from src.teach_competing.crew import TeachCompeting
import os

def run_teach_competing_crew(topic: str) -> str:
    """
    封装 Crew 执行逻辑，供 API 调用

    Args:
        topic (str): 用户输入的主题/问题

    Returns:
        str: 最终生成的文章内容（polisher 的输出）
    """
    # 1. 创建 Crew 实例
    crew_instance = TeachCompeting().crew()

    # 2. 执行 kickoff（传入 topic）
    result = crew_instance.kickoff(inputs={"topic": topic})

    # 3. 读取最终输出文件（polisher 的 output_file）
    output_path = "article2.md"
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            final_article = f.read()
        return final_article
    except FileNotFoundError:
        # 如果文件没生成，返回 raw result 作为 fallback
        return str(result)