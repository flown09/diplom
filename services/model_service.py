import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ModelService:
    REVIEW_THRESHOLD = 0.12
    FAKE_THRESHOLD = 0.50

    CHUNK_SIZE = 190
    CHUNK_STRIDE = 64

    def __init__(self, model_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch.set_num_threads(1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        raw_id2label = getattr(self.model.config, "id2label", None) or {}
        self.id2label = {int(k): v for k, v in raw_id2label.items()}

        # Ожидается: 0 = REAL, 1 = FAKE
        self.real_idx = self._find_label_index({"real", "true", "truth", "not_fake"}, default=0)
        self.fake_idx = self._find_label_index({"fake", "false", "lie"}, default=1)

        self.cls_token_id = getattr(self.tokenizer, "cls_token_id", None)
        self.sep_token_id = getattr(self.tokenizer, "sep_token_id", None)

    def _find_label_index(self, names: set[str], default: int) -> int:
        for idx, label in self.id2label.items():
            if str(label).lower() in names:
                return idx
        return default

    def _chunk_text_by_tokens(self, text: str) -> list[list[int]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]

        if not input_ids:
            return [[]]

        chunks = []
        start = 0

        while start < len(input_ids):
            end = start + self.CHUNK_SIZE
            chunk_ids = input_ids[start:end]
            chunks.append(chunk_ids)

            if end >= len(input_ids):
                break

            start += (self.CHUNK_SIZE - self.CHUNK_STRIDE)

        return chunks

    def _wrap_with_special_tokens(self, chunk_ids: list[int]) -> list[int]:
        input_ids = []

        if self.cls_token_id is not None:
            input_ids.append(self.cls_token_id)

        input_ids.extend(chunk_ids)

        if self.sep_token_id is not None:
            input_ids.append(self.sep_token_id)

        return input_ids

    def _extract_lead(self, text: str, max_sentences: int = 2, max_chars: int = 350) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        lead = " ".join(sentences[:max_sentences]).strip()

        if len(lead) > max_chars:
            lead = lead[:max_chars].rsplit(" ", 1)[0]

        return lead

    @torch.no_grad()
    def _predict_single_chunk_ids(self, chunk_ids: list[int]) -> tuple[float, float, dict]:
        if not chunk_ids:
            return 0.5, 0.5, {}

        input_ids = self._wrap_with_special_tokens(chunk_ids)
        attention_mask = [1] * len(input_ids)

        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=self.device),
            "token_type_ids": torch.zeros((1, len(input_ids)), dtype=torch.long, device=self.device),
        }

        logits = self.model(**inputs).logits[0]
        probas = torch.softmax(logits, dim=-1).detach().cpu().numpy().tolist()

        prob_real = float(probas[self.real_idx]) if self.real_idx < len(probas) else 0.0
        prob_fake = float(probas[self.fake_idx]) if self.fake_idx < len(probas) else 0.0

        label_map = {}
        for i, p in enumerate(probas):
            label = self.id2label.get(i, str(i))
            label_map[label] = float(p)

        return prob_real, prob_fake, label_map

    @torch.no_grad()
    def _predict_text_chunks(self, text: str) -> dict:
        text = (text or "").strip()

        if not text:
            return {
                "prob_real": 0.5,
                "prob_fake": 0.5,
                "labels": {"REAL": 0.5, "FAKE": 0.5},
                "chunks_count": 0,
                "chunk_probs_fake": [],
            }

        chunks = self._chunk_text_by_tokens(text)

        chunk_probs_real = []
        chunk_probs_fake = []
        last_label_map = {}

        for chunk_ids in chunks:
            prob_real, prob_fake, label_map = self._predict_single_chunk_ids(chunk_ids)
            chunk_probs_real.append(prob_real)
            chunk_probs_fake.append(prob_fake)
            last_label_map = label_map

        prob_real = float(sum(chunk_probs_real) / len(chunk_probs_real))
        prob_fake = float(sum(chunk_probs_fake) / len(chunk_probs_fake))

        return {
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "labels": last_label_map,
            "chunks_count": len(chunks),
            "chunk_probs_fake": [float(x) for x in chunk_probs_fake],
        }

    def _make_verdict(self, prob_fake: float) -> tuple[str, str]:
        if prob_fake < self.REVIEW_THRESHOLD:
            return "likely_real", "Вероятно правдивая"
        elif prob_fake < self.FAKE_THRESHOLD:
            return "suspicious", "Подозрительная, нужна проверка"
        else:
            return "likely_fake", "Вероятно фейковая"

    @torch.no_grad()
    def predict_news(self, title: str, text: str) -> dict:
        title = (title or "").strip()
        text = (text or "").strip()

        # Главный вход для текущей модели — ЗАГОЛОВОК,
        # потому что модель обучалась именно на заголовках.
        main_input = title if title else self._extract_lead(text)

        main_result = self._predict_text_chunks(main_input)

        # Дополнительно можно посмотреть лид текста,
        # но не использовать его как основной сигнал.
        lead_text = self._extract_lead(text)
        lead_result = self._predict_text_chunks(lead_text) if lead_text else None

        prob_real = main_result["prob_real"]
        prob_fake = main_result["prob_fake"]

        verdict_code, verdict_text = self._make_verdict(prob_fake)

        return {
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "verdict_code": verdict_code,
            "verdict_text": verdict_text,
            "review_threshold": self.REVIEW_THRESHOLD,
            "fake_threshold": self.FAKE_THRESHOLD,
            "labels": main_result["labels"],
            "chunks_count": main_result["chunks_count"],
            "chunk_probs_fake": main_result["chunk_probs_fake"],

            "used_text": main_input,
            "used_mode": "title" if title else "lead_text",

            "title_prob_fake": main_result["prob_fake"] if title else None,
            "lead_prob_fake": lead_result["prob_fake"] if lead_result else None,
        }