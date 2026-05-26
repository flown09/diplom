import re
import json
import time
import random
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from playwright.sync_api import sync_playwright


BASE_URL = "https://74.ru"

# Сколько нормальных новостей нужно получить
DESIRED_ARTICLES = 100

# Сколько ссылок собрать с запасом
LINKS_TO_COLLECT = 250

OUTPUT_FILE = "dataset_74ru.csv"

# Если сайт часто блокирует headless-режим, поставь False
HEADLESS = True

# Обычные новости вида:
# /text/category/2026/05/13/12345678/
NEWS_URL_RE = re.compile(
    r"^/text/[^/]+/\d{4}/\d{2}/\d{2}/\d+/?$"
)

BAD_TEXT_PARTS = [
    "поделиться",
    "читайте также",
    "по теме",
    "если вы стали очевидцем",
    "сообщить новость",
    "подписаться на новости",
    "реклама",
    "промокод",
    "комментарии",
    "все новости",
    "новости партнеров",
    "нашли ошибку",
    "выделите фрагмент",
]


def normalize_url(href: str) -> str:
    parsed = urlparse(urljoin(BASE_URL, href))
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def is_news_url(href: str) -> bool:
    if not href:
        return False

    full_url = urljoin(BASE_URL, href)
    parsed = urlparse(full_url)

    if not parsed.netloc.endswith("74.ru"):
        return False

    if "/comments/" in parsed.path:
        return False

    # Longread часто имеет другую структуру, лучше пропустить
    if "/text/longread/" in parsed.path:
        return False

    return bool(NEWS_URL_RE.match(parsed.path))


def block_heavy_resources(route):
    resource_type = route.request.resource_type

    if resource_type in ["image", "media", "font", "stylesheet"]:
        route.abort()
    else:
        route.continue_()


def get_html(page, url: str) -> str:
    """
    Быстрая загрузка страницы.
    Не ждем полной загрузки рекламы, картинок и лишних скриптов.
    """
    try:
        page.goto(url, wait_until="commit", timeout=20000)
    except Exception as e:
        raise RuntimeError(f"Не удалось начать загрузку страницы: {e}")

    try:
        page.wait_for_selector("h1, article, main, body", timeout=8000)
    except Exception:
        pass

    try:
        page.evaluate("window.stop()")
    except Exception:
        pass

    page.wait_for_timeout(random.randint(800, 1500))

    try:
        body_text = page.locator("body").inner_text(timeout=8000).lower()
    except Exception:
        body_text = ""

    if "captcha" in body_text or "я не робот" in body_text:
        raise RuntimeError("Похоже, сайт показал капчу")

    return page.content()


def collect_article_links(page, limit: int) -> list[str]:
    links = []
    seen = set()

    list_pages = ["https://74.ru/text/"]
    list_pages += [f"https://74.ru/text/page-{i}/" for i in range(2, 80)]

    for list_url in tqdm(list_pages, desc="Собираю ссылки"):
        if len(links) >= limit:
            break

        try:
            html = get_html(page, list_url)
        except Exception as e:
            print(f"Не удалось открыть страницу ленты {list_url}: {e}")
            continue

        soup = BeautifulSoup(html, "lxml")

        for a in soup.find_all("a", href=True):
            href = a.get("href")

            if is_news_url(href):
                url = normalize_url(href)

                if url not in seen:
                    seen.add(url)
                    links.append(url)

                    if len(links) >= limit:
                        break

        time.sleep(random.uniform(0.7, 1.4))

    return links


def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def is_good_paragraph(text: str) -> bool:
    text = clean_text(text)

    if len(text) < 40:
        return False

    lower = text.lower()

    if any(bad in lower for bad in BAD_TEXT_PARTS):
        return False

    return True


def extract_from_json_ld(soup: BeautifulSoup):
    """
    Иногда текст статьи лежит в JSON-LD.
    Это самый надежный вариант, если он есть.
    """
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        raw = script.string or script.get_text(strip=True)

        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        items = []

        if isinstance(data, dict):
            items.append(data)

            graph = data.get("@graph")
            if isinstance(graph, list):
                items.extend(graph)

        elif isinstance(data, list):
            items.extend(data)

        for item in items:
            if not isinstance(item, dict):
                continue

            article_type = item.get("@type", "")

            if isinstance(article_type, list):
                article_type = " ".join(article_type)

            is_article = (
                "Article" in article_type
                or "NewsArticle" in article_type
                or "ReportageNewsArticle" in article_type
            )

            if not is_article:
                continue

            title = item.get("headline") or item.get("name") or ""
            text = item.get("articleBody") or ""

            if title and text and len(text) > 300:
                return clean_text(title), clean_text(text)

    return None, None


def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        title = clean_text(h1.get_text(" ", strip=True))
        if title:
            return title

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return clean_text(og_title["content"])

    title_tag = soup.find("title")
    if title_tag:
        title = clean_text(title_tag.get_text(" ", strip=True))
        title = title.replace(" — Новости Челябинска и Челябинской области - 74.ru", "")
        title = title.replace(" | 74.ru", "")
        return title

    return ""


def extract_text_by_selectors(soup: BeautifulSoup) -> str:
    """
    Запасной способ, если JSON-LD не дал articleBody.
    """
    selectors = [
        '[itemprop="articleBody"]',
        "article",
        "main",
        '[class*="article"]',
        '[class*="Article"]',
        '[class*="material"]',
        '[class*="Material"]',
        '[data-qa*="article"]',
        '[data-testid*="article"]',
    ]

    paragraphs = []

    for selector in selectors:
        containers = soup.select(selector)

        for container in containers:
            for tag in container.find_all(["p", "h2"], recursive=True):
                piece = clean_text(tag.get_text(" ", strip=True))

                if is_good_paragraph(piece):
                    paragraphs.append(piece)

        if len(paragraphs) >= 3:
            break

    # Последний запасной вариант — все p на странице
    if len(paragraphs) < 3:
        for tag in soup.find_all("p"):
            piece = clean_text(tag.get_text(" ", strip=True))

            if is_good_paragraph(piece):
                paragraphs.append(piece)

    unique = []
    seen = set()

    for p in paragraphs:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return "\n".join(unique)


def extract_article(html: str):
    soup = BeautifulSoup(html, "lxml")

    title, text = extract_from_json_ld(soup)

    if title and text:
        return title, text

    title = extract_title(soup)
    text = extract_text_by_selectors(soup)

    return title, text


def save_rows(rows: list[dict]):
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["url"])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


def main():
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="ru-RU",
            viewport={"width": 1366, "height": 768},
        )

        context.route("**/*", block_heavy_resources)

        page = context.new_page()

        links = collect_article_links(page, LINKS_TO_COLLECT)

        print(f"\nНайдено ссылок: {len(links)}")
        print(f"Нужно собрать статей: {DESIRED_ARTICLES}\n")

        for url in tqdm(links, desc="Собираю статьи"):
            if len(rows) >= DESIRED_ARTICLES:
                break

            try:
                html = get_html(page, url)
                title, text = extract_article(html)

                if title and text and len(text) > 300:
                    row = {
                        "url": url,
                        "title": title,
                        "text": text,
                    }

                    rows.append(row)
                    save_rows(rows)

                    print(f"\nСохранено: {len(rows)} / {DESIRED_ARTICLES}")
                    print(f"Заголовок: {title[:100]}")

                else:
                    print(f"\nПустой или слишком короткий текст: {url}")

                time.sleep(random.uniform(0.8, 1.8))

            except Exception as e:
                print(f"\nОшибка на {url}: {e}")
                continue

        browser.close()

    save_rows(rows)

    print("\nГотово.")
    print(f"Собрано статей: {len(rows)}")
    print(f"Файл сохранен: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()