import re
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ModelService:
    REVIEW_THRESHOLD = 0.37
    FAKE_THRESHOLD = 0.50

    # Максимум для BERT/RuBERT обычно 512 токенов.
    MAX_MODEL_TOKENS = 512

    # Перекрытие между кусками текста.
    # Нужно, чтобы смысл на границе chunks не терялся.
    CHUNK_STRIDE = 128

    # Вес заголовка и полного текста в итоговой оценке.
    # Так как модель обучалась в основном на заголовках,
    # заголовок оставляем важным, но текст делаем главным сигналом.
    TITLE_WEIGHT = 0.35
    TEXT_WEIGHT = 0.65

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

        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", self.MAX_MODEL_TOKENS)

        # У некоторых токенизаторов model_max_length бывает огромным техническим числом.
        if not isinstance(tokenizer_max_length, int) or tokenizer_max_length > 10000:
            tokenizer_max_length = self.MAX_MODEL_TOKENS

        self.max_input_tokens = min(self.MAX_MODEL_TOKENS, tokenizer_max_length)

        special_tokens_count = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.chunk_size = self.max_input_tokens - special_tokens_count

        if self.chunk_size <= 0:
            self.chunk_size = 510

        self.chunk_stride = min(self.CHUNK_STRIDE, max(1, self.chunk_size // 2))

    def _find_label_index(self, names: set[str], default: int) -> int:
        for idx, label in self.id2label.items():
            if str(label).lower() in names:
                return idx
        return default

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _chunk_text_by_tokens(self, text: str) -> list[list[int]]:
        text = self._normalize_text(text)

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )

        input_ids = encoded["input_ids"]

        if not input_ids:
            return []

        chunks = []
        start = 0

        while start < len(input_ids):
            end = start + self.chunk_size
            chunk_ids = input_ids[start:end]
            chunks.append(chunk_ids)

            if end >= len(input_ids):
                break

            start += self.chunk_size - self.chunk_stride

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
        text = self._normalize_text(text)

        if not text:
            return ""

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
        }

        # Для BERT/RuBERT token_type_ids нужны.
        # Для некоторых моделей они не используются, поэтому добавляем только если токенизатор их ожидает.
        if "token_type_ids" in getattr(self.tokenizer, "model_input_names", []):
            inputs["token_type_ids"] = torch.zeros(
                (1, len(input_ids)),
                dtype=torch.long,
                device=self.device
            )

        logits = self.model(**inputs).logits[0]
        probas = torch.softmax(logits, dim=-1).detach().cpu().numpy().tolist()

        prob_real = float(probas[self.real_idx]) if self.real_idx < len(probas) else 0.0
        prob_fake = float(probas[self.fake_idx]) if self.fake_idx < len(probas) else 0.0

        label_map = {}
        for i, p in enumerate(probas):
            label = self.id2label.get(i, str(i))
            label_map[label] = float(p)

        return prob_real, prob_fake, label_map

    def _aggregate_chunk_probs(self, chunk_probs_fake: list[float]) -> float:
        """
        Агрегация вероятностей по chunks.

        mean — слишком мягкий вариант: фейковый абзац может потеряться.
        max — слишком жесткий вариант: один шумный chunk может испортить оценку.

        Поэтому используем смесь:
        70% max + 30% mean.
        Для модерации это обычно лучше: если хотя бы один фрагмент подозрительный,
        новость должна попасть на проверку.
        """
        if not chunk_probs_fake:
            return 0.5

        mean_fake = sum(chunk_probs_fake) / len(chunk_probs_fake)
        max_fake = max(chunk_probs_fake)

        return float(0.70 * max_fake + 0.30 * mean_fake)

    @torch.no_grad()
    def _predict_text_chunks(self, text: str) -> dict:
        text = self._normalize_text(text)

        if not text:
            return {
                "prob_real": 0.5,
                "prob_fake": 0.5,
                "labels": {"REAL": 0.5, "FAKE": 0.5},
                "chunks_count": 0,
                "chunk_probs_fake": [],
                "mean_prob_fake": 0.5,
                "max_prob_fake": 0.5,
            }

        chunks = self._chunk_text_by_tokens(text)

        if not chunks:
            return {
                "prob_real": 0.5,
                "prob_fake": 0.5,
                "labels": {"REAL": 0.5, "FAKE": 0.5},
                "chunks_count": 0,
                "chunk_probs_fake": [],
                "mean_prob_fake": 0.5,
                "max_prob_fake": 0.5,
            }

        chunk_probs_fake = []
        last_label_map = {}

        for chunk_ids in chunks:
            _, prob_fake, label_map = self._predict_single_chunk_ids(chunk_ids)
            chunk_probs_fake.append(prob_fake)
            last_label_map = label_map

        mean_prob_fake = float(sum(chunk_probs_fake) / len(chunk_probs_fake))
        max_prob_fake = float(max(chunk_probs_fake))

        prob_fake = self._aggregate_chunk_probs(chunk_probs_fake)
        prob_real = 1.0 - prob_fake

        return {
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "labels": last_label_map,
            "chunks_count": len(chunks),
            "chunk_probs_fake": [float(x) for x in chunk_probs_fake],
            "mean_prob_fake": mean_prob_fake,
            "max_prob_fake": max_prob_fake,
        }

    def _combine_title_and_text_scores(
        self,
        title_result: dict | None,
        text_result: dict | None,
    ) -> tuple[float, float]:
        has_title = title_result is not None
        has_text = text_result is not None

        if has_title and has_text:
            title_fake = float(title_result["prob_fake"])
            text_fake = float(text_result["prob_fake"])

            prob_fake = (
                self.TITLE_WEIGHT * title_fake
                + self.TEXT_WEIGHT * text_fake
            )

            prob_fake = max(0.0, min(1.0, prob_fake))
            prob_real = 1.0 - prob_fake

            return prob_real, prob_fake

        if has_title:
            prob_fake = float(title_result["prob_fake"])
            return 1.0 - prob_fake, prob_fake

        if has_text:
            prob_fake = float(text_result["prob_fake"])
            return 1.0 - prob_fake, prob_fake

        return 0.5, 0.5

    def _make_verdict(self, prob_fake: float) -> tuple[str, str]:
        if prob_fake < self.REVIEW_THRESHOLD:
            return "likely_real", "Вероятно правдивая"
        elif prob_fake < self.FAKE_THRESHOLD:
            return "suspicious", "Подозрительная, нужна проверка"
        else:
            return "likely_fake", "Вероятно фейковая"

    @torch.no_grad()
    def predict_news(self, title: str, text: str) -> dict:
        title = self._normalize_text(title)
        text = self._normalize_text(text)

        title_result = self._predict_text_chunks(title) if title else None
        text_result = self._predict_text_chunks(text) if text else None

        # Если текста нет, но есть заголовок — работаем по заголовку.
        # Если заголовка нет, но есть текст — работаем по тексту.
        # Если есть и то, и другое — объединяем оценки.
        prob_real, prob_fake = self._combine_title_and_text_scores(
            title_result=title_result,
            text_result=text_result,
        )

        verdict_code, verdict_text = self._make_verdict(prob_fake)

        title_prob_fake = title_result["prob_fake"] if title_result else None
        text_prob_fake = text_result["prob_fake"] if text_result else None

        text_chunks_count = text_result["chunks_count"] if text_result else 0
        text_chunk_probs_fake = text_result["chunk_probs_fake"] if text_result else []

        return {
            "prob_real": prob_real,
            "prob_fake": prob_fake,

            "verdict_code": verdict_code,
            "verdict_text": verdict_text,

            "review_threshold": self.REVIEW_THRESHOLD,
            "fake_threshold": self.FAKE_THRESHOLD,

            "labels": text_result["labels"] if text_result else (title_result["labels"] if title_result else {}),

            # Теперь chunks относятся именно к полному тексту новости.
            "chunks_count": text_chunks_count,
            "chunk_probs_fake": text_chunk_probs_fake,

            "title_prob_fake": title_prob_fake,
            "text_prob_fake": text_prob_fake,

            "text_mean_prob_fake": text_result["mean_prob_fake"] if text_result else None,
            "text_max_prob_fake": text_result["max_prob_fake"] if text_result else None,

            "used_mode": "title_and_full_text" if title and text else "title" if title else "full_text",

            # Для отладки можно посмотреть, что именно использовалось.
            "used_text": {
                "title": title,
                "text_preview": text[:500],
            },
        }