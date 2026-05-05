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
        root.addLayout(right, 5)

        self.title_lbl = QLabel("Заголовок")
        self.title_lbl.setStyleSheet("font-size: 18px; font-weight: 600;")
        right.addWidget(self.title_lbl)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        right.addWidget(self.text, 1)

        self.rules_lbl = QLabel(
            "Правила модели:\n"
            "• P(fake) < 0.12 — вероятно правдивая\n"
            "• 0.12 ≤ P(fake) < 0.50 — подозрительная, нужна проверка\n"
            "• P(fake) ≥ 0.50 — вероятно фейковая"
        )
        self.rules_lbl.setStyleSheet("color: #555;")
        right.addWidget(self.rules_lbl)

        self.score_lbl = QLabel("Оценка модели: —")
        self.score_lbl.setWordWrap(True)
        self.score_lbl.setStyleSheet("font-size: 15px;")
        right.addWidget(self.score_lbl)

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

    def reload(self, keep_news_id: int | None = None):
        self.list.clear()
        self.items = self.repo.list_pending()

        row_to_select = None

        for i, n in enumerate(self.items):
            label_txt = f" | {n.model_label}" if n.model_label else ""
            score_txt = f" | P(fake)={n.model_score:.2f}" if n.model_score is not None else ""
            self.list.addItem(f"#{n.id}: {n.title[:60]}{label_txt}{score_txt}")

            if keep_news_id is not None and n.id == keep_news_id:
                row_to_select = i

        if row_to_select is not None:
            self.list.setCurrentRow(row_to_select)

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
        self.score_lbl.setText("Оценка модели: выполняется анализ...")

        self.worker = PredictWorker(self.model_service, n.id, n.title, n.text)
        self.worker.done.connect(self.on_pred_done)
        self.worker.failed.connect(self.on_pred_failed)
        self.worker.start()

    def on_pred_done(self, news_id: int, result: dict):
        if not self.current_news:
            return

        # Если пользователь уже переключился на другую новость — игнорируем старый результат
        if self.current_news.id != news_id:
            return

        prob_fake = result["prob_fake"]
        prob_real = result["prob_real"]
        verdict_text = result["verdict_text"]
        verdict_code = result["verdict_code"]

        self.repo.update_model_result(news_id, prob_fake, verdict_text)

        icon = self._verdict_icon(verdict_code)
        used_mode = result.get("used_mode", "title")
        title_prob_fake = result.get("title_prob_fake")
        lead_prob_fake = result.get("lead_prob_fake")

        extra = ""
        if title_prob_fake is not None:
            extra += f"\nP(fake) по заголовку = {title_prob_fake:.2f}"
        if lead_prob_fake is not None:
            extra += f"\nP(fake) по первым предложениям = {lead_prob_fake:.2f}"

        self.score_lbl.setText(
            f"{icon} {verdict_text}\n"
            f"P(fake) = {prob_fake:.2f}\n"
            f"P(real) = {prob_real:.2f}\n"
            f"Режим анализа: {used_mode}\n"
            f"Чанков проанализировано: {result.get('chunks_count', 1)}"
            f"{extra}"
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