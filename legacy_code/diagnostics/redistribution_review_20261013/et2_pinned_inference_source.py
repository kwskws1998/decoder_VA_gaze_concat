import torch
import transformers
import numpy as np


FEATURE_NAMES = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
WINDOW_SIZE = 512
OVERLAP = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    from safetensors.torch import load_file as st_load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

class RobertaRegressionModel(torch.nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = transformers.RobertaModel.from_pretrained(model_name)
        embed_size = 1024 if 'large' in model_name else 768
        self.decoder = torch.nn.Linear(embed_size, 5)

    def forward(self, input_ids, attention_mask, predict_mask):
        hidden = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        Y_pred = self.decoder(hidden)
        mask = (predict_mask == 0).unsqueeze(-1).expand_as(Y_pred).to(Y_pred.device)
        Y_pred = Y_pred.masked_fill(mask, -1.0)
        return Y_pred

class FixationsPredictor2:

    def __init__(self, checkpoint_path, model_name='roberta-base'):
        self.model_name = model_name
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(
            model_name, add_prefix_space=True
        )
        self.model = RobertaRegressionModel(model_name).to(device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    def _load_checkpoint(self, path):
        import os
        if path.endswith('.safetensors'):
            if not HAS_SAFETENSORS:
                raise ImportError('pip install safetensors')
            self.model.load_state_dict(st_load_file(path, device=str(device)))
        elif path.endswith('.pt') or path.endswith('.bin'):
            self.model.load_state_dict(torch.load(path, map_location=device))
        else:
            for ext in ['.safetensors', '.pt']:
                if os.path.exists(path + ext):
                    self._load_checkpoint(path + ext)
                    return
            raise FileNotFoundError(f'체크포인트 없음: {path}')

    def _predict_with_sliding_window(self, input_ids_full, attention_mask_full):
        seq_len = input_ids_full.shape[1]

        if seq_len <= WINDOW_SIZE:
            predict_mask = attention_mask_full.clone()
            with torch.no_grad():
                pred = self.model(input_ids_full, attention_mask_full, predict_mask)
            return pred.squeeze(0).cpu().numpy()

        predictions = np.zeros((seq_len, 5), dtype=np.float32)
        weights = np.zeros(seq_len, dtype=np.float32)
        stride = WINDOW_SIZE - OVERLAP

        start = 0
        while start < seq_len:
            end = min(start + WINDOW_SIZE, seq_len)
            ids_win = input_ids_full[:, start:end]
            mask_win = attention_mask_full[:, start:end]
            predict_mask = mask_win.clone()

            with torch.no_grad():
                pred_win = self.model(ids_win, mask_win, predict_mask)
            pred_np = pred_win.squeeze(0).cpu().numpy()

            win_len = end - start
            linear_w = np.ones(win_len, dtype=np.float32)
            if start > 0:
                ramp_len = min(OVERLAP, win_len)
                linear_w[:ramp_len] = np.linspace(0, 1, ramp_len)
            if end < seq_len:
                ramp_len = min(OVERLAP, win_len)
                linear_w[-ramp_len:] = np.linspace(1, 0, ramp_len)

            for feat_i in range(5):
                predictions[start:end, feat_i] += pred_np[:, feat_i] * linear_w
            weights[start:end] += linear_w

            if end == seq_len:
                break
            start += stride

        nonzero = weights > 0
        predictions[nonzero] /= weights[nonzero, None]
        return predictions

    def _get_word_boundaries(self, input_ids):
        tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in input_ids]
        words = []
        current_word_tokens = []
        current_indices = []

        for i, tok in enumerate(tokens):
            if tok in ('<s>', '</s>', '<pad>'):
                if current_word_tokens:
                    words.append(current_indices)
                    current_word_tokens = []
                    current_indices = []
                continue

            if tok.startswith('Ġ') or not current_word_tokens:
                if current_word_tokens:
                    words.append(current_indices)
                current_word_tokens = [tok]
                current_indices = [i]
            else:
                current_word_tokens.append(tok)
                current_indices.append(i)

        if current_word_tokens:
            words.append(current_indices)

        return words

    def predict_raw_text(self, text):
        words_list = text.strip().split()
        encoding = self.tokenizer(
            [words_list],
            is_split_into_words=True,
            return_tensors='pt',
            truncation=False,
            padding=False,
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        token_preds = self._predict_with_sliding_window(input_ids, attention_mask)

        word_boundaries = self._get_word_boundaries(input_ids.squeeze(0).cpu().tolist())

        word_features = np.zeros((len(word_boundaries), 5), dtype=np.float32)
        for w_idx, token_indices in enumerate(word_boundaries):
            first_tok = token_indices[0]
            pred = token_preds[first_tok]
            pred = np.clip(pred, 0, None)
            word_features[w_idx] = pred

        return word_features, words_list

    def predict_and_remap_to_tokenizer(self, input_ids_rm, attention_mask_rm, rm_tokenizer):
        batch_size = input_ids_rm.shape[0]
        seq_len_rm = input_ids_rm.shape[1]

        fixations_batch = []
        masks_batch = []

        for b in range(batch_size):
            ids = input_ids_rm[b].cpu().tolist()
            mask = attention_mask_rm[b].cpu().tolist()

            pad_id = rm_tokenizer.pad_token_id
            ids_no_pad = [i for i, m in zip(ids, mask) if m == 1 and i != pad_id]
            text = rm_tokenizer.decode(ids_no_pad, skip_special_tokens=True)

            word_features, _ = self.predict_raw_text(text)

            remapped = self._remap_features_to_rm_tokens(
                word_features, text, ids, mask, rm_tokenizer
            )

            fixations_batch.append(remapped)
            masks_batch.append(torch.tensor(mask, dtype=torch.long))

        fixations = torch.stack(fixations_batch)
        fixations_attention_mask = torch.stack(masks_batch)

        return fixations, fixations_attention_mask

    def _compute_mapped_fixations(self, input_ids_rm, attention_mask_rm=None):
        # gaze_reward reward_model_base.py의 fixations_model_version=2 호환 인터페이스
        if attention_mask_rm is None:
            attention_mask_rm = torch.ones_like(input_ids_rm)
        ids = input_ids_rm[0].cpu().tolist()
        mask = attention_mask_rm[0].cpu().tolist()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 1
        ids_no_pad = [i for i, m in zip(ids, mask) if m == 1 and i != pad_id]
        text = self.tokenizer.decode(ids_no_pad, skip_special_tokens=True)
        word_features, _ = self.predict_raw_text(text)
        remapped = self._remap_features_to_rm_tokens(
            word_features, text, ids, mask, self.tokenizer
        )
        fixations = remapped.unsqueeze(0)
        fix_attn = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
        return fixations, fix_attn, None, None, None, None

    def _remap_features_to_rm_tokens(self, word_features, text, rm_input_ids, rm_mask, rm_tokenizer):
        words = text.strip().split()
        seq_len = len(rm_input_ids)
        output = torch.zeros(seq_len, 5, dtype=torch.float32)

        rm_tokens = rm_tokenizer.convert_ids_to_tokens(rm_input_ids)

        word_to_rm_indices = _align_words_to_rm_tokens(words, rm_tokens, rm_tokenizer)

        n_words = min(len(words), len(word_features))
        for w_idx in range(n_words):
            if w_idx >= len(word_to_rm_indices):
                break
            indices = word_to_rm_indices[w_idx]
            if not indices:
                continue
            feat = torch.tensor(word_features[w_idx], dtype=torch.float32)
            if indices[0] < seq_len and rm_mask[indices[0]] == 1:
              output[indices[0]] = feat
        return output


def _align_words_to_rm_tokens(words, rm_tokens, rm_tokenizer):
    special_ids = set(rm_tokenizer.all_special_ids)
    word_to_indices = []
    tok_idx = 0

    for word in words:
        indices = []
        chars_remaining = len(word)

        while tok_idx < len(rm_tokens) and chars_remaining > 0:
            tok = rm_tokens[tok_idx]
            tok_id = rm_tokenizer.convert_tokens_to_ids(tok)
            if tok_id in special_ids:
                tok_idx += 1
                continue

            tok_clean = tok.lstrip('Ġ▁ ')
            indices.append(tok_idx)
            chars_remaining -= len(tok_clean)
            tok_idx += 1

        word_to_indices.append(indices)

    return word_to_indices
