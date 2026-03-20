import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelService:
    def __init__(self, model_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        # Пытаемся понять названия классов
        self.id2label = getattr(self.model.config, "id2label", None) or {}

    @torch.no_grad()
    def predict_truth_proba(self, text: str, max_length: int = 256) -> tuple[float, dict]:
        """
        Возвращает:
        - probability_of_truth (float)
        - debug dict (labels/probas)
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits[0]
        probas = torch.softmax(logits, dim=-1).detach().cpu().numpy().tolist()

        # Собираем отладочную карту label->proba
        label_map = {}
        for i, p in enumerate(probas):
            label = self.id2label.get(i, str(i))
            label_map[label] = float(p)

        # Как выбрать "правда"?
        # Вариант A: если в id2label есть TRUTH/REAL
        truth_keys = [k for k in label_map.keys() if k.lower() in ("real", "true", "truth", "not_fake")]
        if truth_keys:
            truth_proba = label_map[truth_keys[0]]
        else:
            # Вариант B (fallback): считаем, что класс 1 = правда
            # (лучше один раз подогнать по твоей конфигурации)
            truth_proba = float(probas[1]) if len(probas) > 1 else float(probas[0])

        return truth_proba, {"labels": label_map}
