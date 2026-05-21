import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass
class News:
    id: int
    title: str
    text: str
    created_at: str
    status: str
    model_score: float | None
    model_label: str | None
    source_url: str | None = None
    contact: str | None = None


def _get_chelyabinsk_tz() -> timezone | ZoneInfo:
    try:
        return ZoneInfo("Asia/Yekaterinburg")
    except ZoneInfoNotFoundError:
        # Fallback for environments without IANA tz database (e.g. Windows without tzdata).
        return timezone(timedelta(hours=5))


CHELYABINSK_TZ = _get_chelyabinsk_tz()


def _now_chelyabinsk_iso() -> str:
    return datetime.now(CHELYABINSK_TZ).isoformat()


class Repo:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                source_url TEXT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                model_score REAL,
                model_label TEXT,
                contact TEXT
            );

            CREATE TABLE IF NOT EXISTS moderation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_id INTEGER NOT NULL,
                decision TEXT NOT NULL,
                comment TEXT,
                moderator TEXT,
                decided_at TEXT NOT NULL,
                FOREIGN KEY(news_id) REFERENCES news(id)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO settings(key, value) VALUES ('threshold', '0.12');
            """)

            columns = [row[1] for row in con.execute("PRAGMA table_info(news)").fetchall()]
            if "contact" not in columns:
                con.execute("ALTER TABLE news ADD COLUMN contact TEXT")

            # Для теста — добавим новости, если база пустая
            cur = con.execute("SELECT COUNT(*) FROM news")
            if cur.fetchone()[0] == 0:
                now = _now_chelyabinsk_iso()
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("Тестовая новость 1", "В Москве открыли новый парк...", now)
                )
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("Тестовая новость 2", "Срочно! Шок! Невероятное событие...", now)
                )
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("Тестовая новость 3", "В России полностью запретили интернет.", now)
                )
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("Тестовая новость 4", "Путин запретил дышать чиновником воздухом. Теперь они дышат газами зеков.", now)
                )
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("Тестовая новость 5", "Путин запретил интернет в России", now)
                )
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("Тестовая новость 6", "МОСКВА, 16 фев — РИА Новости. Полтора десятка человек спасли из горящего административного здания на севере Москвы, рассказали РИА Новости в пресс-службе МЧС. \"Из горящего здания пожарными спасены 16 человек\", — сказали в ведомстве. Шестерых из них осмотрели медики. В здании по адресу Магистральный тупик, дом 10, корпус 1 горел четвертый этаж. Площадь пожара составила 80 квадратных метров, открытое горение уже ликвидировали. Как передает корреспондент РИА Новости, из окон четвертого этажа свисают перевязанные простыни. Предположительно, по ним люди спускались на этаж ниже, спасаясь от огня. Столичный главк Следственного комитета возбудил уголовное дело. На месте работают следователи и криминалисты.", now)
                )
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("Тестовая новость 7", "на площади Тяньаньмэнь ничего не произошло", now)
                )
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("В Казани открыли новый детский сад на 220 мест.", "В одном из новых микрорайонов Казани завершилось строительство детского сада на 220 мест.", now)
                )
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("В Новосибирске из-за сильного снегопада задержали несколько рейсов.", "В аэропорту Новосибирска утром были задержаны несколько внутренних рейсов из-за сильного снегопада и ухудшения видимости.", now)
                )
                con.execute(
                    "INSERT INTO news(title, text, created_at) VALUES (?, ?, ?)",
                    ("В школе Екатеринбурга прошла олимпиада по математике.", "В одной из школ Екатеринбурга состоялся городской этап олимпиады по математике среди старшеклассников.", now)
                )

    def add_news(self, title: str, text: str, source_url: str | None = None, contact: str | None = None) -> int:
        now = _now_chelyabinsk_iso()
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO news(title, text, source_url, contact, created_at, status) VALUES (?, ?, ?, ?, ?, 'pending')",
                (title, text, source_url, contact, now),
            )
            return int(cur.lastrowid)

    def list_pending(self) -> list[News]:
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT id, title, text, created_at, status, model_score, model_label, source_url, contact
                FROM news
                WHERE status='pending'
                ORDER BY id DESC
                """
            ).fetchall()
        return [News(*r) for r in rows]

    def update_model_result(self, news_id: int, score: float, label: str):
        with self._conn() as con:
            con.execute(
                "UPDATE news SET model_score=?, model_label=? WHERE id=?",
                (score, label, news_id)
            )

    def set_decision(self, news_id: int, decision: str, comment: str = "", moderator: str = "moderator"):
        ts = _now_chelyabinsk_iso()
        with self._conn() as con:
            status = "approved" if decision == "approve" else "rejected" if decision == "reject" else "pending"
            con.execute("UPDATE news SET status=? WHERE id=?", (status, news_id))
            con.execute(
                "INSERT INTO moderation_log(news_id, decision, comment, moderator, decided_at) VALUES (?, ?, ?, ?, ?)",
                (news_id, decision, comment, moderator, ts)
            )

    def list_public_news(self, page: int = 1, per_page: int = 10) -> tuple[list[News], int]:
        page = max(page, 1)
        offset = (page - 1) * per_page
        with self._conn() as con:
            total_count = con.execute(
                "SELECT COUNT(*) FROM news WHERE status='approved'"
            ).fetchone()[0]
            rows = con.execute(
                """
                SELECT n.id,
                       n.title,
                       n.text,
                       COALESCE(
                           (
                               SELECT ml.decided_at
                               FROM moderation_log AS ml
                               WHERE ml.news_id = n.id AND ml.decision = 'approve'
                               ORDER BY ml.id DESC
                               LIMIT 1
                           ),
                           n.created_at
                       ) AS created_at,
                       n.status,
                       n.model_score,
                       n.model_label,
                       n.source_url,
                       n.contact
                FROM news AS n
                WHERE n.status='approved'
                ORDER BY n.id DESC
                LIMIT ? OFFSET ?
                """
                ,
                (per_page, offset),
            ).fetchall()
        total_pages = max((total_count + per_page - 1) // per_page, 1)
        return [News(*r) for r in rows], total_pages

    def get_public_news_by_id(self, news_id: int) -> News | None:
        with self._conn() as con:
            row = con.execute(
                """
                SELECT n.id,
                       n.title,
                       n.text,
                       COALESCE(
                           (
                               SELECT ml.decided_at
                               FROM moderation_log AS ml
                               WHERE ml.news_id = n.id AND ml.decision = 'approve'
                               ORDER BY ml.id DESC
                               LIMIT 1
                           ),
                           n.created_at
                       ) AS created_at,
                       n.status,
                       n.model_score,
                       n.model_label,
                       n.source_url,
                       n.contact
                FROM news AS n
                WHERE n.id=? AND n.status='approved'
                """,
                (news_id,),
            ).fetchone()
        return News(*row) if row else None
