from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from data.repo import Repo

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = str(BASE_DIR / "moderator.sqlite3")

app = FastAPI(title="News Site")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

repo = Repo(DB_PATH)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    news = repo.list_public_news()
    return templates.TemplateResponse("index.html", {"request": request, "news": news})


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
        news = repo.list_public_news()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "errors": errors,
                "title": title,
                "text": text,
                "source_url": source_url or "",
                "contact": contact,
                "news": news,
                "show_modal": True,
            },
            status_code=400,
        )

    news_id = repo.add_news(title=title, text=text, source_url=source_url, contact=contact)
    return templates.TemplateResponse("success.html", {"request": request, "news_id": news_id})
