from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QTextEdit,
    QLabel, QPushButton, QMessageBox
)


class PredictWorker(QThread):
    done = Signal(int, object)
    failed = Signal(str)

    def __init__(self, model_service, news_id: int, title: str, text: str):
        super().__init__()
        self.model_service = model_service
        self.news_id = news_id
        self.title = title
        self.text = text

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            result = self.model_service.predict_news(self.title, self.text)
            if not self.isInterruptionRequested():
                self.done.emit(self.news_id, result)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QWidget):
    TITLE_PREVIEW_LIMIT = 25

    def __init__(self, repo, model_service):
        super().__init__()
        self.repo = repo
        self.model_service = model_service
        self.current_news = None
        self.worker = None
        self.items = []

        self.setWindowTitle("Модератор новостей")
        self.resize(1100, 700)

        root = QHBoxLayout(self)

        # Слева список
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.on_select)
        root.addWidget(self.list, 2)

        # Справа детали
        right = QVBoxLayout()
        root.addLayout(right, 6)

        top = QHBoxLayout()
        self.title_lbl = QLabel("Заголовок")
        self.title_lbl.setStyleSheet("font-size: 18px; font-weight: 600;")
        top.addWidget(self.title_lbl, 1)

        self.refresh_btn = QPushButton("Обновить список")
        self.refresh_btn.clicked.connect(self.on_refresh_clicked)
        top.addWidget(self.refresh_btn)

        self.rules_btn = QPushButton("Правила модели")
        self.rules_btn.clicked.connect(self.show_rules)
        top.addWidget(self.rules_btn)
        right.addLayout(top)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("font-size: 16px; line-height: 1.45;")
        self.text.setMaximumHeight(300)
        right.addWidget(self.text, 1)

        self.score_lbl = QLabel("Оценка модели: —")
        self.score_lbl.setWordWrap(True)
        self.score_lbl.setStyleSheet("font-size: 15px;")
        self.score_lbl.setMaximumHeight(95)
        right.addWidget(self.score_lbl, 0)

        # Кнопки решения
        btns = QHBoxLayout()
        self.approve_btn = QPushButton("Одобрить")
        self.reject_btn = QPushButton("Отклонить")
        self.approve_btn.clicked.connect(lambda: self.make_decision("approve"))
        self.reject_btn.clicked.connect(lambda: self.make_decision("reject"))
        btns.addWidget(self.approve_btn)
        btns.addWidget(self.reject_btn)
        right.addLayout(btns)

        self.reload()

    def _truncate_title(self, title: str) -> str:
        title = (title or "").strip()
        if len(title) <= self.TITLE_PREVIEW_LIMIT:
            return title
        return f"{title[: self.TITLE_PREVIEW_LIMIT - 1]}…"

    def _list_item_text(self, n) -> str:
        title = self._truncate_title(n.title)
        if n.model_score is not None:
            return f"#{n.id}: {title} | {n.model_label} | P(fake)={n.model_score:.2f}"
        return f"#{n.id}: {title} | оценка: не выполнена"

    def reload(self, keep_news_id: int | None = None):
        self.list.clear()
        self.items = self.repo.list_pending()

        row_to_select = None

        for i, n in enumerate(self.items):
            self.list.addItem(self._list_item_text(n))
            if keep_news_id is not None and n.id == keep_news_id:
                row_to_select = i

        if row_to_select is not None:
            self.list.setCurrentRow(row_to_select)


    def on_refresh_clicked(self):
        keep_news_id = self.current_news.id if self.current_news else None
        self.reload(keep_news_id=keep_news_id)

    def show_rules(self):
        QMessageBox.information(
            self,
            "Правила модели",
            "• P(fake) < 0.12 — вероятно правдивая\n"
            "• 0.12 ≤ P(fake) < 0.50 — подозрительная, нужна проверка\n"
            "• P(fake) ≥ 0.50 — вероятно фейковая"
        )

    def on_select(self, idx: int):
        if idx < 0 or idx >= len(self.items):
            self.current_news = None
            self.title_lbl.setText("Заголовок")
            self.text.clear()
            self.score_lbl.setText("Оценка модели: —")
            return

        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()

        n = self.items[idx]
        self.current_news = n
        self.title_lbl.setText(n.title)
        self.text.setText(n.text)

        if n.model_score is not None and n.model_label:
            self.score_lbl.setText(f"{n.model_label}\nP(fake) = {n.model_score:.2f}")
            return

        self.score_lbl.setText("Оценка модели: выполняется анализ...")

        self.worker = PredictWorker(self.model_service, n.id, n.title, n.text)
        self.worker.done.connect(self.on_pred_done)
        self.worker.failed.connect(self.on_pred_failed)
        self.worker.start()

    def on_pred_done(self, news_id: int, result: dict):
        if not self.current_news:
            return

        if self.current_news.id != news_id:
            return

        prob_fake = result["prob_fake"]
        prob_real = result["prob_real"]
        verdict_text = result["verdict_text"]
        verdict_code = result["verdict_code"]

        self.repo.update_model_result(news_id, prob_fake, verdict_text)

        icon = self._verdict_icon(verdict_code)
        self.score_lbl.setText(
            f"{icon} {verdict_text}\n"
            f"P(fake) = {prob_fake:.2f}\n"
            f"P(real) = {prob_real:.2f}"
        )

        self.reload(keep_news_id=news_id)

    def on_pred_failed(self, message: str):
        QMessageBox.critical(self, "Ошибка модели", message)
        self.score_lbl.setText("Оценка модели: ошибка")

    def _verdict_icon(self, code: str) -> str:
        if code == "likely_real":
            return "✅"
        if code == "suspicious":
            return "⚠️"
        if code == "likely_fake":
            return "❌"
        return "•"

    def make_decision(self, decision: str):
        if not self.current_news:
            return

        self.repo.set_decision(
            self.current_news.id,
            decision,
            comment="",
            moderator="moderator"
        )

        QMessageBox.information(self, "Готово", "Решение сохранено.")
        self.current_news = None
        self.title_lbl.setText("Заголовок")
        self.text.clear()
        self.score_lbl.setText("Оценка модели: —")
        self.reload()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        super().closeEvent(event)
