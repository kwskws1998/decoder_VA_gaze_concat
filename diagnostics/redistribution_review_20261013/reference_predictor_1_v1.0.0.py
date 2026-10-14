import torch
import os
import torch.nn as nn
import pathlib
from transformers import AutoTokenizer
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
from eyetrackpy.data_generator.models.fixations_aligner import FixationsAligner
from collections.abc import Iterable
import re
from eyetrackpy.data_generator.fixations_predictor.models.model_manager import download_model

class BiLSTMRegression(nn.Module):
    def __init__(self, embedding, hidden_dim, drop_out) -> None:
        super().__init__()
        self.emb = embedding
        self.emb.requires_grad_(False)

        self.lstm = nn.LSTM(
            input_size=self.emb.weight.size(1),
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=drop_out,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.emb(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x.squeeze(-1)


import threading


class FixationsPredictor_1:
    def __init__(
        self,
        hidden_dim,
        drop_out=0.2,
        modelTokenizer=None,
        remap=True,
    ) -> None:
        vocab_size = 32128
        embedding = nn.Embedding(vocab_size, 512)
        self.model = BiLSTMRegression(embedding, hidden_dim, drop_out)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # fixTokenizer.pad_token = fixTokenizer.eos_token
        # modelTokenizer.pad_token = modelTokenizer.eos_token
        self.modelTokenizer = modelTokenizer
        self.tokenizer_lock = threading.Lock()  # DummyContext()
        self.remap = remap

        self.fixTokenizer = AutoTokenizer.from_pretrained(
            "t5-small", cache_dir="./cache/models", model_max_length=2048
        )
        try:
            path = str(
                pathlib.Path(__file__)
                .parent.resolve()
                .parent.resolve()
                .parent.resolve()
            )
            FP_dir = os.path.join(
                path,
                "fixations_predictor_trained_1",
                # "FPmodels",
                "T5-tokenizer-BiLSTM-TRT-12-concat-3",
            )
            if not os.path.isfile(FP_dir):
                download_model('model_1')

            self.model.load_state_dict(torch.load(FP_dir))
        except:
            path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
            FP_dir = os.path.join(
                path,
                "fixations_predictor_trained_1",
                # "FPmodels",
                "T5-tokenizer-BiLSTM-TRT-12-concat-3",
            )
            self.model.load_state_dict(torch.load(FP_dir))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _compute_mapped_fixations(
        self, input_ids_original=None, sentences=None, return_all=False
    ):
        # we destokenize the sentences with the model tokenizer (since is the one with  we have tokenized)
        # then we tokenize the sentences with the fixations tokenizer and the model one again
        # part of this process is to remove special tokens (like chat ones) and then retokenize without them
        if isinstance(input_ids_original, Iterable) and not isinstance(
            input_ids_original, torch.Tensor
        ):
            if "input_ids" in input_ids_original:
                input_ids_original = input_ids_original["input_ids"]

        if sentences is None and input_ids_original is None:
            raise ValueError("sentences or input_ids_original must be provided")
        if input_ids_original is not None and self.modelTokenizer is None:
            raise ValueError(
                "modelTokenizer must be provided i you provide input_ids_original"
            )
        if self.remap and self.modelTokenizer is None:
            raise ValueError("modelTokenizer must be provided if remap is True")

        if sentences is None:
            pattern = r"(user|assistant)\r?\n"
            device = input_ids_original.device
            sentences = []
            for s_idx in range(input_ids_original.shape[0]):
                with self.tokenizer_lock:
                    sentence = self.modelTokenizer.decode(
                        input_ids_original[s_idx], skip_special_tokens=True
                    )

                # TODO: decide if we use this or not
                sentence = re.sub(pattern, "", sentence)
                sentence = sentence.strip()
                sentences.append(sentence)
            # if len(sentences) == 1:
            #     sentences = sentences[0]
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if isinstance(sentences, str):
                sentences = [sentences]
        with self.tokenizer_lock:
            if self.remap:
                text_tokenized_model = self.modelTokenizer(
                    sentences,
                    padding=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            text_tokenized_fix = self.fixTokenizer(
                sentences,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

        # --------------------------------------------------------------------------------------------
        # compute fixations
        fixations = self.model.forward(text_tokenized_fix["input_ids"].to(device))
        # fixations2 = self.model.forward(text_tokenized_fix["input_ids"].to(device))
        # fixations3 = self.model.forward(text_tokenized_fix["input_ids"].to(device))
        # test1 = fixations == fixations2
        # test1 = torch.sum(test1 == False).item()
        # test2 = fixations == fixations3
        # test2 = torch.sum(test2 == False).item()
        # test3 = fixations3 == fixations2
        # test3 = torch.sum(test3 == False).item()
        if self.remap is False:
            fixations = fixations.to(device)
            fixations_attention_mask = text_tokenized_fix["attention_mask"].to(device)
            return (
                fixations,
                fixations_attention_mask,
                None,
                None,
                text_tokenized_fix,
                sentences,
            )
        else:
            fixations = fixations.tolist()
            if not return_all:
                # map tokens between original model tokenizer and fixations tokenizer
                # we have the fixations on the fixations tokenizer, we need to map them to the model tokenizer
                tokens_id_mapped = TokenizerAligner().align_tokens(
                    sentences,
                    text_tokenized_model,
                    text_tokenized_fix,
                    return_all=False,
                )
                mapped_fixations = (
                    FixationsAligner.map_fixations_between_tokens_correct(
                        fixations,
                        tokens_id_mapped,
                        input_ids_original,
                        text_tokenized_model,
                        text_tokenized_fix,
                        return_all=False,
                    )
                )
                if len(mapped_fixations[0]) != input_ids_original[0].shape[0]:
                    print(
                        f"mapped_fixations ({len(mapped_fixations[0])}), input_ids_original ({input_ids_original[0].shape[0]}), text_tokenized_model ({text_tokenized_model[0].shape[0]})"
                    )
                # --------------------------------------------------------------------------------------------
                mapped_fixations = torch.FloatTensor(mapped_fixations).to(device)
                return (
                    fixations,
                    None,
                    mapped_fixations,
                    text_tokenized_model,
                    text_tokenized_fix,
                    sentences,
                )
            tokens_idx_mapped, tokens_id_mapped, words_str_mapped, tokens_str_mapped = (
                TokenizerAligner().align_tokens(sentences, text_tokenized_model, text_tokenized_fix, return_all=True)
            )
            mapped_fixations_corrected, mapped_fixations = (
                FixationsAligner.map_fixations_between_tokens_correct(
                    fixations,
                    tokens_id_mapped,
                    input_ids_original,
                    text_tokenized_model,
                    text_tokenized_fix,
                    return_all=True,
                )
            )
            return (
                fixations,
                None,
                mapped_fixations_corrected,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
                tokens_idx_mapped,
                tokens_id_mapped,
                words_str_mapped,
                tokens_str_mapped,
            )

    def forward(self, input_ids_original=None, sentences=None):
        (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            model_tok,
            fix_tok,
            sentences,
        ) = self._compute_mapped_fixations(input_ids_original, sentences=sentences)
        if self.remap:
            return mapped_fixations  # ['input_ids'].cuda()

        else:
            return (
                fixations,
                fixations_attention_mask,
            )
