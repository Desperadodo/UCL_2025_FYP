# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from sacrebleu.metrics import BLEU
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    from nltk.translate.meteor_score import meteor_score  # type: ignore


if is_rouge_available():
    from rouge_score import rouge_scorer  # type: ignore


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            if self.all_preds:
                bleu_scores = BLEU(tokenize="13a", lowercase=False, effective_order=False).corpus_score(self.all_preds, [self.all_labels])
                p = [p / 100 for p in bleu_scores.precisions]
                bp = bleu_scores.bp

                # Cumulative BLEU-1
                result["bleu-1"] = bp * p[0] * 100 if p[0] > 0 else 0.0

                # Cumulative BLEU-2
                if p[0] > 0 and p[1] > 0:
                    result["bleu-2"] = bp * np.exp(0.5 * (np.log(p[0]) + np.log(p[1]))) * 100
                else:
                    result["bleu-2"] = 0.0

                # Cumulative BLEU-3
                if p[0] > 0 and p[1] > 0 and p[2] > 0:
                    result["bleu-3"] = bp * np.exp((1 / 3) * (np.log(p[0]) + np.log(p[1]) + np.log(p[2]))) * 100
                else:
                    result["bleu-3"] = 0.0

                # Cumulative BLEU-4 is the standard score
                result["bleu-4"] = bleu_scores.score

        self.score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "meteor": [],
        }
        self.all_preds = []
        self.all_labels = []
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        self.all_preds.extend(decoded_preds)
        self.all_labels.extend(decoded_labels)

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = pred.split()
            reference = label.split()

            if len(hypothesis) == 0 or len(reference) == 0:
                result = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            else:
                scores = scorer.score(label, pred)
                result = {k: scores[k].fmeasure for k in scores}

            self.score_dict["rouge-1"].append(round(result["rouge1"] * 100, 4))
            self.score_dict["rouge-2"].append(round(result["rouge2"] * 100, 4))
            self.score_dict["rouge-l"].append(round(result["rougeL"] * 100, 4))

            self.score_dict["meteor"].append(round(meteor_score([reference], hypothesis) * 100, 4))

        if compute_result:
            return self._dump()
