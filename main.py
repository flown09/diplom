import os
import sys
from pathlib import Path

# Для стабильности на Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from PySide6.QtWidgets import QApplication

from data.repo import Repo
from services.model_service import ModelService
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    base_dir = Path(__file__).resolve().parent
    db_path = str(base_dir / "moderator.sqlite3")
    model_dir = str(base_dir / "model")

    repo = Repo(db_path)

    # Для десктопного приложения на Windows лучше CPU:
    # чуть медленнее, зато заметно стабильнее.
    model = ModelService(model_dir=model_dir, device="cpu")

    w = MainWindow(repo, model)
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()