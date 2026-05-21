from pathlib import Path
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from data.repo import Repo

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = str(BASE_DIR / "moderator.sqlite3")

app = FastAPI(title="News Site")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

repo = Repo(DB_PATH)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, page: int = 1):
    news, total_pages = repo.list_public_news(page=page)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"news": news, "page": page, "total_pages": total_pages},
    )


@app.get("/news/{news_id}", response_class=HTMLResponse)
def news_detail(request: Request, news_id: int):
    item = repo.get_public_news_by_id(news_id)
    if not item:
        raise HTTPException(status_code=404, detail="Новость не найдена")

    return templates.TemplateResponse(
        request=request,
        name="news_detail.html",
        context={"item": item},
    )


@app.post("/submit", response_class=HTMLResponse)
def submit(
    request: Request,
    title: str = Form(...),
    text: str = Form(...),
    source_url: str = Form(""),
    contact: str = Form(""),
):
    title = title.strip()
    text = text.strip()
    source_url = source_url.strip() or None
    contact = contact.strip()

    errors = []

    if not title:
        errors.append("Заголовок обязателен.")

    if len(text) < 30:
        errors.append("Текст слишком короткий (минимум 30 символов).")

    if not contact:
        errors.append("Укажите телефон или email для связи.")

    if errors:
        news, total_pages = repo.list_public_news(page=1)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "errors": errors,
                "title": title,
                "text": text,
                "source_url": source_url or "",
                "contact": contact,
                "news": news,
                "page": 1,
                "total_pages": total_pages,
                "show_modal": True,
            },
            status_code=400,
        )

    news_id = repo.add_news(
        title=title,
        text=text,
        source_url=source_url,
        contact=contact,
    )

    return templates.TemplateResponse(
        request=request,
        name="success.html",
        context={"news_id": news_id},
    )
