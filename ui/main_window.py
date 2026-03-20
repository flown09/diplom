from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QTextEdit,
    QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QFormLayout, QMessageBox
)

class PredictWorker(QThread):
    done = Signal(float)

    def __init__(self, model_service, text: str):
        super().__init__()
        self.model_service = model_service
        self.text = text

    def run(self):
        score, _debug = self.model_service.predict_truth_proba(self.text)
        self.done.emit(score)

class MainWindow(QWidget):
    def __init__(self, repo, model_service):
        super().__init__()
        self.repo = repo
        self.model_service = model_service
        self.current_news = None
        self.worker = None

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

        self.score_lbl = QLabel("Оценка модели: —")
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

        # Настройки
        gb = QGroupBox("Настройки проверки")
        form = QFormLayout(gb)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(self.repo.get_threshold())
        self.threshold.valueChanged.connect(self.on_threshold_change)
        form.addRow("Порог правдивости:", self.threshold)
        right.addWidget(gb)

        self.reload()

    def reload(self):
        self.list.clear()
        self.items = self.repo.list_pending()
        for n in self.items:
            score_txt = f" | P={n.model_score:.2f}" if n.model_score is not None else ""
            self.list.addItem(f"#{n.id}: {n.title[:60]}{score_txt}")

    def on_select(self, idx: int):
        if idx < 0 or idx >= len(self.items):
            return
        n = self.items[idx]
        self.current_news = n
        self.title_lbl.setText(n.title)
        self.text.setText(n.text)
        self.score_lbl.setText("Оценка модели: считаю...")

        # фоновой инференс
        self.worker = PredictWorker(self.model_service, n.text)
        self.worker.done.connect(self.on_pred_done)
        self.worker.start()

    def on_pred_done(self, score: float):
        if not self.current_news:
            return
        self.repo.update_model_score(self.current_news.id, score)
        thr = self.threshold.value()
        flag = "✅" if score >= thr else "⚠️"
        self.score_lbl.setText(f"Оценка модели: {flag} P(правда)={score:.2f} (порог {thr:.2f})")
        self.reload()

    def make_decision(self, decision: str):
        if not self.current_news:
            return
        self.repo.set_decision(self.current_news.id, decision, comment="", moderator="moderator")
        QMessageBox.information(self, "Готово", "Решение сохранено.")
        self.current_news = None
        self.title_lbl.setText("Заголовок")
        self.text.clear()
        self.score_lbl.setText("Оценка модели: —")
        self.reload()

    def on_threshold_change(self, v: float):
        self.repo.set_threshold(float(v))
