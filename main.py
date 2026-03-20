import sys
from PySide6.QtWidgets import QApplication
from data.repo import Repo
from pathlib import Path
from services.model_service import ModelService
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)

    BASE_DIR = Path(__file__).resolve().parent
    DB_PATH = str(BASE_DIR / "moderator.sqlite3")
    MODEL_DIR = str(BASE_DIR / "model")  # папка model рядом с main.py

    repo = Repo(DB_PATH)
    model = ModelService(model_dir=MODEL_DIR)

    w = MainWindow(repo, model)
    w.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
