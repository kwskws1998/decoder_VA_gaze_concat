import torch
import transformers
import sys
import pathlib
import os
sys.path.append("../..")
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
import threading
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
from eyetrackpy.data_generator.models.fixations_aligner import FixationsAligner
from data_generator.fixations_predictor.models.model_manager import download_model
import re

device = torch.device("cuda")


class RobertaRegressionModel(torch.nn.Module):
    def __init__(self, model_name="roberta-base"):
        super(RobertaRegressionModel, self).__init__()

        if "roberta" in model_name:
            self.roberta = transformers.RobertaModel.from_pretrained(
                model_name, device_map="auto"
            )
        elif "bert" in model_name:
            self.roberta = transformers.BertModel.from_pretrained(
                model_name, device_map="auto"
            )

        EMBED_SIZE = 1024 if "large" in model_name else 768
        self.max_length = 512
        self.overlap = 50
        self.decoder = torch.nn.Sequential(torch.nn.Linear(EMBED_SIZE, 5)).to(
            self.roberta.device
        )

    def _forward(self, X_ids, X_attns, predict_mask=None):
        """
        X_ids: (B, seqlen) tensor of token ids
        X_attns: (B, seqlen) tensor of attention masks, 0 for [PAD] tokens and 1 otherwise
        predict_mask: (B, seqlen) tensor, 1 for tokens that we need to predict

        Output: (B, seqlen, 5) tensor of predictions, only predict when predict_mask == 1
        """

        # (B, seqlen, 768)
        temp = self.roberta(X_ids, attention_mask=X_attns).last_hidden_state

        # (B, seqlen, 5)
        Y_pred = self.decoder(temp)
        if predict_mask is not None:
            # Where predict_mask == 0, set Y_pred to -1
            # we change this to zero, because the model was trainned predicting only the first token of each word
            # we add here zeros because later we are going to add the predictions from all tokens of the word
            # Y_pred[predict_mask == 0] = -1
            if isinstance(predict_mask, list):
                predict_mask = torch.tensor(predict_mask)
            Y_pred[predict_mask == 0] = 0

        return Y_pred

    def forward(self, X_ids, X_attns, predict_mask=None):
        if X_ids.shape[1] > self.max_length:
            return self._forward_sliding_window(
                X_ids, X_attns, predict_mask, overlap=self.overlap
            )
        else:
            return self._forward(X_ids, X_attns, predict_mask)

    def _forward_sliding_window(self, X_ids, X_attns, predict_mask=None, overlap=50):
        """
        Sliding Window Approach:
        This refers to the method where a fixed-size window (or chunk) slides over the input sequence with a certain overlap
        """
        # Create chunks from input tensor and attention mask
        chunks, attention_chunks, predict_mask_chunks = self.create_chunks_batch(
            X_ids, X_attns, predict_mask, max_lenght=self.max_length, overlap=overlap
        )
        # Process each chunk through the model (B, chunk_dimension, 5)
        batch_chunk_outputs = []
        for chunks, attention_chunks, predict_mask_chunks in zip(
            chunks, attention_chunks, predict_mask_chunks
        ):
            output = self._forward(
                X_ids=chunks, X_attns=attention_chunks, predict_mask=predict_mask_chunks
            )
            batch_chunk_outputs.append(output)
        # recombine chunk outputs to have all sequence output
        combined_output = self.linear_approximation_output_overlapping_window(
            batch_chunk_outputs, overlap=overlap
        )

        return combined_output

    @staticmethod
    def create_chunks_batch(
        input_tensors, attention_masks, predict_masks=None, max_lenght=512, overlap=50
    ):
        batch_chunks = []
        batch_attention_chunks = []
        batch_mask_chunks = []
        for i in range(len(input_tensors)):  # Process each batch element separately
            input_tensor = input_tensors[i]
            attention_mask = attention_masks[i]
            if predict_masks is not None:
                predict_mask = predict_masks[i]

            chunks = []
            attention_chunks = []
            mask_chunks = []
            start = 0
            while start < len(input_tensor):
                end = min(start + max_lenght, len(input_tensor))
                chunks.append(input_tensor[start:end])
                attention_chunks.append(attention_mask[start:end])
                if predict_masks is not None:
                    mask_chunks.append(predict_mask[start:end])
                if end == len(input_tensor):
                    break
                start = end - overlap  # Move start to account for overlap

            batch_chunks.append(chunks)
            batch_attention_chunks.append(attention_chunks)
            if predict_masks is not None:
                batch_mask_chunks.append(mask_chunks)

        # batch_chunks_reorder = []
        # batch_attention_chunks_reorder = []
        # batch_mask_chunks_reorder = []
        # for i in list(range(len(batch_chunks))):
        #     for j in range(len(batch_chunks[i])):
        #       if i==0:
        #         batch_chunks_reorder.append([batch_chunks[i][j]])
        #         batch_attention_chunks_reorder.append([batch_attention_chunks[i][j]])
        #         if predict_masks is not None:
        #           batch_mask_chunks_reorder.append([torch.tensor(batch_mask_chunks[i][j])])
        #       else:
        #         batch_chunks_reorder[j].append(batch_chunks[i][j])
        #         batch_attention_chunks_reorder[j].append(batch_attention_chunks[i][j])
        #         if predict_masks is not None:
        #           batch_mask_chunks_reorder[j].append(torch.tensor(batch_mask_chunks[i][j]))
        # for i in list(range(len(batch_chunks_reorder))):
        #     batch_chunks_reorder[i] = torch.stack(batch_chunks_reorder[i])
        #     batch_attention_chunks_reorder[i] = torch.stack(batch_attention_chunks_reorder[i])
        #     if predict_masks is not None:
        #       batch_mask_chunks_reorder[i] = torch.stack(batch_mask_chunks_reorder[i])
        batch_chunks_reorder = [[] for _ in range(len(batch_chunks[0]))]
        batch_attention_chunks_reorder = [[] for _ in range(len(batch_chunks[0]))]
        batch_mask_chunks_reorder = (
            [[] for _ in range(len(batch_chunks[0]))]
            if predict_masks is not None
            else None
        )

        # Efficiently reorder the chunks using list comprehension
        for i in range(len(batch_chunks)):
            for j in range(len(batch_chunks[i])):
                batch_chunks_reorder[j].append(batch_chunks[i][j])
                batch_attention_chunks_reorder[j].append(batch_attention_chunks[i][j])
                if predict_masks is not None:
                    batch_mask_chunks_reorder[j].append(
                        torch.tensor(batch_mask_chunks[i][j])
                    )

        # Convert lists to tensors if needed
        batch_chunks_reorder = [torch.stack(chunk) for chunk in batch_chunks_reorder]
        batch_attention_chunks_reorder = [
            torch.stack(chunk) for chunk in batch_attention_chunks_reorder
        ]
        if predict_masks is not None:
            batch_mask_chunks_reorder = [
                torch.stack(chunk) for chunk in batch_mask_chunks_reorder
            ]
        else:
            batch_mask_chunks_reorder = None
        return (
            batch_chunks_reorder,
            batch_attention_chunks_reorder,
            batch_mask_chunks_reorder,
        )

    @staticmethod
    def linear_approximation_output_overlapping_window(batch_outputs, overlap):
        batch_size = batch_outputs[0].shape[0]  # Get the batch size
        final_outputs = []
        # Process each batch element independently
        for batch_index in range(batch_size):
            combined_output = batch_outputs[0][
                batch_index
            ]  # Start with the first chunk's output for this batch element

            # Iterate through each output chunk (except the first one)
            for chunk_idx in range(1, len(batch_outputs)):
                current_output = batch_outputs[chunk_idx][batch_index]
                overlap_start = combined_output.shape[0] - overlap

                # Perform linear interpolation over the overlapping region for all output values
                for j in range(overlap):
                    if (
                        overlap_start + j >= combined_output.shape[0]
                        or j >= current_output.shape[0]
                    ):
                        break

                    alpha = (j + 1) / (overlap + 1)  # Linear weight
                    # Apply linear approximation for all values (num_outputs)
                    combined_output[overlap_start + j] = (1 - alpha) * combined_output[
                        overlap_start + j
                    ] + alpha * current_output[j]

                # Append the non-overlapping part
                combined_output = torch.cat(
                    (combined_output, current_output[overlap:]), dim=0
                )

            final_outputs.append(combined_output)

        # Stack all outputs together to form a final batch tensor
        final_outputs = torch.stack(final_outputs)

        return final_outputs


class FixationsPredictor_2:
    def __init__(
        self,
        model_name="roberta-base",
        model_path=str(pathlib.Path(__file__).parent.resolve()) + "/model.pth",
        modelTokenizer=None,
        remap=True,
    ):
        self.model_name = model_name
        # device = torch.device('cuda')
        # self.model = RobertaRegressionModel(model_name).to(device)
        self.model = RobertaRegressionModel(model_name)
        #check if model_path is a file
        if not os.path.isfile(model_path):
            download_model('model_2')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.fixTokenizer = transformers.RobertaTokenizerFast.from_pretrained(
            model_name, add_prefix_space=True
        )
        self.modelTokenizer = modelTokenizer
        self.remap = remap
        self.tokenizer_lock = threading.Lock()

    def _compute_mapped_fixations(
        self, input_ids_original=None, attention_mask_original=None, sentences=None
    ):
        # we destokenize the sentences with the model tokenizer (since is the one with  we have tokenized )
        # then we tokenize the sentences with the fixations tokenizer and the model one again
        # part of this process is to remove special tokens (like chat ones) and then retokenize without them

        if sentences is None and input_ids_original is None:
            raise ValueError("sentences or input_ids_original must be provided")
        if input_ids_original is not None and self.modelTokenizer is None:
            raise ValueError(
                "modelTokenizer must be provided i you provide input_ids_original"
            )
        if self.remap and self.modelTokenizer is None:
            raise ValueError("modelTokenizer must be provided if remap is True")
        # this undoing of the tokenization is to be able to tokenize the sentences with the fixations tokenizer, to remap and to compute first_token_word
        # also we undo the aply chat and remove the user, assistant,
        if input_ids_original is not None:
            pattern = r"(user|assistant)\r?\n"
            device = input_ids_original.device
            sentences = []
            for s_idx in range(input_ids_original.shape[0]):
                with self.tokenizer_lock:
                    sentence = self.modelTokenizer.decode(
                        input_ids_original[s_idx], skip_special_tokens=True
                    )
                sentence = re.sub(pattern, "", sentence)
                sentence = sentence.strip()
                sentences.append(sentence)
            # if len(sentences) == 1:
            #     sentences = sentences[0]
        elif sentences is not None:
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
            # we dont truncate because we are usin sliding window overlapping later in roberta
            text_tokenized_fix = self.fixTokenizer(
                sentences,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
                # truncation=True,
            )

        # --------------------------------------------------------------------------------------------
        # compute fixations
        if input_ids_original is not None:
            input_ids_original = input_ids_original.detach()
        X_ids = text_tokenized_fix["input_ids"].to(device)
        fixations_attention_mask = text_tokenized_fix["attention_mask"].to(device)
        # we compute first_token_word because the model was trained to predict only for the first token of each word
        first_token_word = TokenizerAligner().search_first_token_word(
            text_tokenized_fix
        )
        fixations = self.model.forward(
            X_ids, X_attns=fixations_attention_mask, predict_mask=first_token_word
        )
        # -----------code to test if the model is working correctly and the predictions are the same------
        # fixations2 = self.model.forward(X_ids, X_attns=fixations_attention_mask, predict_mask=first_token_word)
        # fixations3 = self.model.forward(X_ids, X_attns=fixations_attention_mask, predict_mask=first_token_word)
        # test1 = fixations == fixations2
        # test1 = torch.sum(test1 == False).item()
        # test2 = fixations == fixations3
        # test2 = torch.sum(test2 == False).item()
        # test3 = fixations3 == fixations2
        # test3 = torch.sum(test3 == False).item()
        # -----------<end> code to test if the model is working correctly and the predictions are the same------
        X_ids = X_ids.detach()

        if self.remap is False:
            fixations = fixations.to(device)
            return (
                fixations,
                fixations_attention_mask,
                None,
                None,
                text_tokenized_fix,
                sentences,
            )
        else:
            # map tokens between original model tokenizer and fixations tokenizer
            # we have the fixations on the fixations tokenizer, we need to map them to the model tokenizer
            tokens_id_mapped = TokenizerAligner().align_tokens(
                sentences, text_tokenized_model, text_tokenized_fix, return_all=False
            )
            fixations_features = torch.split(fixations, 1, dim=-1)
            fixations_features_mapped = []
            for i in range(len(fixations_features)):
                fix_mapped = FixationsAligner.map_fixations_between_tokens_correct(
                    fixations_features[i].squeeze(),
                    tokens_id_mapped,
                    input_ids_original,
                    text_tokenized_model,
                    text_tokenized_fix,
                    return_all=False,
                )
                fix_mapped = torch.FloatTensor(fix_mapped).to(device).unsqueeze(-1)
                fixations_features_mapped.append(fix_mapped)

            # --------------------------------------------------------------------------------------------
            mapped_fixations = torch.cat(fixations_features_mapped, dim=2)
            if mapped_fixations.shape[1] != input_ids_original[0].shape[0]:
                print(
                    f"mapped_fixations ({mapped_fixations.shape[1]}), input_ids_original ({input_ids_original[0].shape[0]}), text_tokenized_model ({text_tokenized_model[0].shape[0]})"
                )
            return (
                fixations,
                None,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
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

    # def predict(self, valid_df):
    #   valid_data = src.dataloader.EyeTrackingCSV(valid_df, model_name=self.model_name)
    #   valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=16)

    #   predict_df = valid_df.copy()
    #   predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = 9999

    #   # Assume one-to-one matching between nonzero predictions and tokens
    #   predictions = []
    #   self.model.eval()
    #   for _, X_ids, X_attns, Y_true in valid_loader:
    #     token_predictions = self._predict(X_ids, X_attns, Y_true)
    #     predictions.append(token_predictions)

    #   predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = np.vstack(predictions)
    #   return predict_df

    # def _predict(self, X_ids, X_attns, Y_true):
    #   X_ids = X_ids.to(device)
    #   X_attns = X_attns.to(device)
    #   predict_mask = torch.sum(Y_true, axis=2) >= 0
    #   with torch.no_grad():
    #     Y_pred = self.model(X_ids, X_attns, predict_mask).cpu()

    #   for batch_ix in range(X_ids.shape[0]):
    #     for row_ix in range(X_ids.shape[1]):
    #       token_prediction = Y_pred[batch_ix, row_ix]
    #       if token_prediction.sum() != -5.0:
    #         token_prediction[token_prediction < 0] = 0

    #   return token_prediction
