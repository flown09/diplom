from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from data.repo import Repo

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = str(BASE_DIR / "moderator.sqlite3")

app = FastAPI(title="News Site")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

repo = Repo(DB_PATH)


RU_MONTHS = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def format_pub_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    return f"{dt.day} {RU_MONTHS[dt.month - 1]} {dt:%H:%M}"


templates.env.filters["format_pub_date"] = format_pub_date


@app.get("/", response_class=HTMLResponse)
def index(request: Request, page: int = 1, submitted: int = 0):
    news, total_pages = repo.list_public_news(page=page)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "news": news,
            "page": page,
            "total_pages": total_pages,
            "show_submitted_toast": bool(submitted),
        },
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

    repo.add_news(
        title=title,
        text=text,
        source_url=source_url,
        contact=contact,
    )

    return RedirectResponse(url="/?submitted=1", status_code=303)
