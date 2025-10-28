from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import models, database
from database.database import get_db
from pydantic import BaseModel
from .crew_runner import run_teach_competing_crew
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse


templates = Jinja2Templates(directory="templates")

# 创建数据库表（首次运行时）
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="AI 教师文章生成 API")

# Pydantic 模型
class ArticleCreate(BaseModel):
    topic: str

class ArticleResponse(BaseModel):
    id: int
    topic: str
    content: str | None = None
    status: str

    class Config:
        from_attributes = True  # 替代 orm_mode=True（Pydantic v2）

@app.post("/generate", response_model=ArticleResponse)
async def generate(request: ArticleCreate, db: Session = Depends(get_db)):
    # 1. 先保存请求到数据库
    db_request = models.ArticleRequest(
        topic=request.topic,
        status="processing"
    )
    db.add(db_request)
    db.commit()
    db.refresh(db_request)

    try:
        content = run_teach_competing_crew(topic=request.topic)

        # 4. 更新数据库
        db_request.status = "success"
        db_request.content = content
        db.commit()
        db.refresh(db_request)

        return db_request

    except Exception as e:
        db_request.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.get("/article/{id}", response_model=ArticleResponse)
def get_article(id: int, db: Session = Depends(get_db)):
    article = db.query(models.ArticleRequest).filter(models.ArticleRequest.id == id).first()
    if not article:
        raise HTTPException(status_code=404, detail="文章未找到")
    return article

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})