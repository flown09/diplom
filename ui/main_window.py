from html import escape
from urllib.parse import urlparse

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QTextEdit,
    QLabel, QPushButton, QMessageBox, QTabWidget
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
    TITLE_PREVIEW_LIMIT = 20

    def __init__(self, repo, model_service):
        super().__init__()
        self.repo = repo
        self.model_service = model_service
        self.current_news = None
        self.worker = None
        self.items = []

        self.setWindowTitle("Модератор новостей")
        self.resize(1700, 900)
        self._apply_large_font_styles()

        root = QHBoxLayout(self)

        # Слева вкладки со списками
        self.tabs = QTabWidget()
        self.pending_list = QListWidget()
        self.approved_list = QListWidget()
        self.rejected_list = QListWidget()

        self.tabs.addTab(self.pending_list, "На модерации")
        self.tabs.addTab(self.approved_list, "Одобренные")
        self.tabs.addTab(self.rejected_list, "Отклоненные")

        self.pending_list.currentRowChanged.connect(lambda idx: self.on_select("pending", idx))
        self.approved_list.currentRowChanged.connect(lambda idx: self.on_select("approved", idx))
        self.rejected_list.currentRowChanged.connect(lambda idx: self.on_select("rejected", idx))
        self.tabs.currentChanged.connect(self.on_tab_changed)

        root.addWidget(self.tabs, 3)

        # Справа детали
        right = QVBoxLayout()
        root.addLayout(right, 6)

        top = QHBoxLayout()
        self.title_lbl = QLabel("Заголовок")
        self.title_lbl.setStyleSheet("font-size: 38px; font-weight: 700;")
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
        self.text.setStyleSheet("font-size: 30px; line-height: 1.6;")
        self.text.setMaximumHeight(520)
        right.addWidget(self.text, 1)

        score_row = QHBoxLayout()

        self.score_lbl = QLabel("Оценка модели: —")
        self.score_lbl.setWordWrap(True)
        self.score_lbl.setStyleSheet("font-size: 28px;")
        self.score_lbl.setMaximumHeight(280)
        score_row.addWidget(self.score_lbl, 1)

        contact_source_col = QVBoxLayout()
        contact_source_col.setSpacing(8)

        self.contact_lbl = QLabel("Контакт отправителя:\n—")
        self.contact_lbl.setStyleSheet("font-size: 24px; color: #666;")
        self.contact_lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.contact_lbl.setWordWrap(False)
        contact_source_col.addWidget(self.contact_lbl)

        self.source_lbl = QLabel("Источник:\n—")
        self.source_lbl.setStyleSheet("font-size: 24px; color: #666;")
        self.source_lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.source_lbl.setWordWrap(False)
        self.source_lbl.setOpenExternalLinks(True)
        self.source_lbl.setTextFormat(Qt.RichText)
        self.source_lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
        contact_source_col.addWidget(self.source_lbl)

        score_row.addLayout(contact_source_col, 0)

        right.addLayout(score_row, 0)

        # Кнопки решения
        btns = QHBoxLayout()
        self.approve_btn = QPushButton("Одобрить")
        self.reject_btn = QPushButton("Отклонить")
        self.approve_btn.clicked.connect(lambda: self.make_decision("approve"))
        self.reject_btn.clicked.connect(lambda: self.make_decision("reject"))
        btns.addWidget(self.approve_btn)
        btns.addWidget(self.reject_btn)

        self.return_btn = QPushButton("Вернуть на модерацию")
        self.return_btn.clicked.connect(self.return_to_moderation)
        btns.addWidget(self.return_btn)
        right.addLayout(btns)

        self.reload()

    def _apply_large_font_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                font-size: 24px;
            }
            QTabBar::tab {
                font-size: 24px;
                padding: 14px 24px;
            }
            QListWidget {
                font-size: 24px;
            }
            QListWidget::item {
                padding: 12px 8px;
            }
            QPushButton {
                font-size: 24px;
                padding: 14px 24px;
            }
            QMessageBox QLabel {
                font-size: 24px;
            }
            """
        )

    def _truncate_title(self, title: str) -> str:
        title = (title or "").strip()
        if len(title) <= self.TITLE_PREVIEW_LIMIT:
            return title
        return f"{title[: self.TITLE_PREVIEW_LIMIT - 1]}…"

    def _list_item_text(self, n) -> str:
        title = self._truncate_title(n.title)
        if n.model_score is not None:
            return f"{title} | {n.model_label} | P(fake)={n.model_score:.2f}"
        return f"{title} | оценка: не выполнена"

    def _active_status(self) -> str:
        return ["pending", "approved", "rejected"][self.tabs.currentIndex()]

    def _active_list_widget(self) -> QListWidget:
        return [self.pending_list, self.approved_list, self.rejected_list][self.tabs.currentIndex()]

    def _active_status(self) -> str:
        return ["pending", "approved", "rejected"][self.tabs.currentIndex()]

    def _active_list_widget(self) -> QListWidget:
        return [self.pending_list, self.approved_list, self.rejected_list][self.tabs.currentIndex()]

    def reload(self, keep_news_id: int | None = None):
        self.pending_items = self.repo.list_by_status("pending")
        self.approved_items = self.repo.list_by_status("approved")
        self.rejected_items = self.repo.list_by_status("rejected")
        self.items = {
            "pending": self.pending_items,
            "approved": self.approved_items,
            "rejected": self.rejected_items,
        }

        self.pending_list.clear()
        self.approved_list.clear()
        self.rejected_list.clear()

        for n in self.pending_items:
            self.pending_list.addItem(self._list_item_text(n))

        for n in self.approved_items:
            self.approved_list.addItem(self._list_item_text(n))

        for n in self.rejected_items:
            self.rejected_list.addItem(self._list_item_text(n))

        active_status = self._active_status()
        active_items = self.items[active_status]
        row_to_select = next((i for i, n in enumerate(active_items) if n.id == keep_news_id), None)

        if row_to_select is not None:
            self._active_list_widget().setCurrentRow(row_to_select)
        else:
            self._active_list_widget().setCurrentRow(-1)


    def on_refresh_clicked(self):
        keep_news_id = self.current_news.id if self.current_news else None
        self.reload(keep_news_id=keep_news_id)

    def show_rules(self):
        QMessageBox.information(
            self,
            "Правила модели",
            "• P(fake) < 0.37 – вероятно правдивая\n"
            "• 0.37 ≤ P(fake) < 0.50 – подозрительная, нужна проверка\n"
            "• P(fake) ≥ 0.50 – вероятно фейковая"
        )

    def on_tab_changed(self, _: int):
        self.current_news = None
        self.title_lbl.setText("Заголовок")
        self.text.clear()
        self.score_lbl.setText("Оценка модели: –")
        self.contact_lbl.setText("Контакт отправителя:\n–")
        self.source_lbl.setText("Источник:\n–")
        self._active_list_widget().setCurrentRow(-1)
        self._set_action_buttons_state()

    def on_select(self, status: str, idx: int):
        status_items = self.items.get(status, [])

        if idx < 0 or idx >= len(status_items):
            self.current_news = None
            self.title_lbl.setText("Заголовок")
            self.text.clear()
            self.score_lbl.setText("Оценка модели: –")
            self.contact_lbl.setText("Контакт отправителя:\n–")
            self.source_lbl.setText("Источник:\n–")
            self._set_action_buttons_state()
            return

        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()

        n = status_items[idx]
        self.current_news = n
        self.title_lbl.setText(n.title)
        self.text.setText(n.text)
        self._set_action_buttons_state()
        self.contact_lbl.setText(f"Контакт отправителя:\n{n.contact or 'не указан'}")
        self.source_lbl.setText(self._format_source_label(n.source_url))

        if n.model_score is not None and n.model_label:
            self.score_lbl.setText(f"{n.model_label}\nP(fake) = {n.model_score:.2f}")
            return

        self.score_lbl.setText("Оценка модели: выполняется анализ...")

        self.worker = PredictWorker(self.model_service, n.id, n.title, n.text)
        self.worker.done.connect(self.on_pred_done)
        self.worker.failed.connect(self.on_pred_failed)
        self.worker.start()


    def _format_source_label(self, source_url: str | None) -> str:
        if not source_url:
            return "Источник:\nне указан"

        clean_url = source_url.strip()
        if not clean_url:
            return "Источник:\nне указан"

        parsed = urlparse(clean_url)
        href = clean_url if parsed.scheme else f"https://{clean_url}"
        safe_href = escape(href, quote=True)
        safe_text = escape(clean_url)
        return f'Источник:<br><a href="{safe_href}">{safe_text}</a>'

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
        # self.score_lbl.setText(
        #     f"{icon} {verdict_text}\n"
        #     f"P(fake) = {prob_fake:.2f}\n"
        #     f"P(real) = {prob_real:.2f}"
        # )
        title_prob_fake = result.get("title_prob_fake")
        text_prob_fake = result.get("text_prob_fake")
        chunks_count = result.get("chunks_count", 0)

        details = ""

        if title_prob_fake is not None:
            details += f"\nP(fake) по заголовку = {title_prob_fake:.2f}"

        if text_prob_fake is not None:
            details += f"\nP(fake) по тексту = {text_prob_fake:.2f}"
            details += f"\nФрагментов текста: {chunks_count}"

        self.score_lbl.setText(
            f"{icon} {verdict_text}\n"
            f"Итоговая P(fake) = {prob_fake:.2f}\n"
            f"Итоговая P(real) = {prob_real:.2f}"
            f"{details}"
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
        self.contact_lbl.setText("Контакт отправителя:\n—")
        self.source_lbl.setText("Источник:\n—")
        self.reload()

    def return_to_moderation(self):
        if not self.current_news:
            return

        self.repo.return_to_moderation(
            self.current_news.id,
            comment="Возвращено в модерацию",
            moderator="moderator"
        )

        QMessageBox.information(self, "Готово", "Новость возвращена на модерацию.")
        self.current_news = None
        self.title_lbl.setText("Заголовок")
        self.text.clear()
        self.score_lbl.setText("Оценка модели: —")
        self.contact_lbl.setText("Контакт отправителя:\n—")
        self.source_lbl.setText("Источник:\n—")
        self.reload()

    def _set_action_buttons_state(self):
        status = self.current_news.status if self.current_news else None
        is_pending = status == "pending"
        is_resolved = status in ("approved", "rejected")
        self.approve_btn.setEnabled(is_pending)
        self.reject_btn.setEnabled(is_pending)
        self.return_btn.setEnabled(is_resolved)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        super().closeEvent(event)
