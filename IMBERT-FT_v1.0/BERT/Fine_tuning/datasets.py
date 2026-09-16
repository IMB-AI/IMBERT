"""Dataset that turns a pathway abundance-rank table into IMBERT inputs.
Note on in-place mutation
-------------------------
`__getitem__` must not modify the stored sequences: the same dataset object is
reused for every epoch when `num_workers = 0`, so a masked token written back
into the store would leak into later epochs. The original code relied on
DataLoader workers being re-forked each epoch to undo that, which only holds
when `num_workers > 0`. Here the sequence is copied before masking, so every
epoch sees the pristine sample whatever the worker setting.
"""

import random

import torch

from torch.utils.data import Dataset


class IMBDatasetForBinaryClassification(Dataset):
    def __init__(self,
                 inhouse_input_df,
                 vocab_dict,
                 sample_name_id_dict,
                 disease_sample_list,
                 seq_len,
                 add_special_token):

        self.vocab_dict = vocab_dict
        self.seq_len = seq_len
        self.df = inhouse_input_df
        self.sample_name_id_dict = sample_name_id_dict
        self.disease_sample_list = disease_sample_list
        self.add_special_token = add_special_token

        # Pathways missing from the vocabulary are encoded as [UNK] instead of
        # raising, so a table built with a newer pathway database still works.
        self.unk_id = vocab_dict['[UNK]']
        self.mask_id = vocab_dict['[MASK]']
        self.pad_id = vocab_dict['[PAD]']
        self.min_token_id = min(vocab_dict.values())
        self.max_token_id = max(vocab_dict.values())

        self.sample_name_list = [sample for sample in inhouse_input_df.columns.to_list()
                                 if sample != 'Unnamed: 0']

        self.total_sample_ids_list = []
        self.sample_match_input_dict = {}
        for sample in self.sample_name_list:
            tmp_df = inhouse_input_df.loc[:, ["Unnamed: 0", sample]]
            non_zero_tmp_df = tmp_df.loc[tmp_df[sample] > 0, :].sort_values(by=sample, ascending=True)
            pwy_ids_by_rank = [vocab_dict.get(pwy_name, self.unk_id)
                               for pwy_name in non_zero_tmp_df["Unnamed: 0"]]

            self.total_sample_ids_list.append(pwy_ids_by_rank)
            self.sample_match_input_dict[sample] = pwy_ids_by_rank

    def __getitem__(self, index):
        sample_pwy_ids_list_with_rand_masking, masking_label, true_label, sample_name_id_list, disease_label_list = self._random_masking(
            index)

        true_label = self._handle_sentence_for_length(true_label)
        sample_pwy_ids_list_with_rand_masking = self._handle_sentence_for_length(sample_pwy_ids_list_with_rand_masking)
        masking_label = self._handle_sentence_for_length(masking_label)

        if self.add_special_token:
            bert_answer = [self.vocab_dict['[CLS]']] + true_label + [self.vocab_dict['[SEP]']]
            bert_input = [self.vocab_dict['[CLS]']] + sample_pwy_ids_list_with_rand_masking + [self.vocab_dict['[SEP]']]
            bert_label = [self.vocab_dict['[PAD]']] + masking_label + [
                self.vocab_dict['[PAD]']]  # because of special token
        else:
            bert_answer = true_label
            bert_input = sample_pwy_ids_list_with_rand_masking
            bert_label = masking_label

        segment_label = [0 for _ in range(len(bert_input))]
        attention_mask = [1 for _ in range(len(bert_input))]

        padding = self._padding(bert_input)
        sample_id_padding = self._padding(sample_name_id_list)
        disease_label_padding = self._padding(disease_label_list)

        bert_input.extend(padding)
        bert_label.extend(padding)
        segment_label.extend(padding)
        bert_answer.extend(padding)
        attention_mask.extend(padding)
        sample_name_id_list.extend(sample_id_padding)
        disease_label_list.extend(disease_label_padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "bert_answer": bert_answer,
                  "sample_name_id": sample_name_id_list,
                  "disease_label": disease_label_list,
                  "attention_mask": attention_mask}

        return {key: torch.tensor(value) for key, value in output.items()}

    def __len__(self):
        return len(self.total_sample_ids_list)

    def _padding(self, object_list):
        return [self.vocab_dict['[PAD]'] for _ in range(self.seq_len - len(object_list))]

    def _handle_sentence_for_length(self, object_list):
        if self.add_special_token:
            final_seq_len = self.seq_len - 2  # [CLS] and [SEP]
            if len(object_list) > final_seq_len:
                len_fitted_obj_list = object_list[:final_seq_len]
            else:
                len_fitted_obj_list = object_list
        else:
            final_seq_len = self.seq_len
            if len(object_list) > final_seq_len:
                len_fitted_obj_list = object_list[:final_seq_len]
            else:
                len_fitted_obj_list = object_list

        return len_fitted_obj_list

    def _random_masking(self, index):
        output_label = []
        true_label = []

        # Copy: the stored sequence must survive this epoch untouched.
        sample_pwy_id_list = list(self.total_sample_ids_list[index])
        sample_name = self.sample_name_list[index]
        sample_name_id_list = [self.sample_name_id_dict[sample_name]]
        disease_label_list = [1 if sample_name in self.disease_sample_list else 0]

        for i, token in enumerate(sample_pwy_id_list):
            true_label.append(token)
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    sample_pwy_id_list[i] = self.mask_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    sample_pwy_id_list[i] = random.randrange(self.min_token_id,
                                                            self.max_token_id + 1)

                # 10% randomly change token to current token
                else:
                    sample_pwy_id_list[i] = token

                output_label.append(token)

            else:
                sample_pwy_id_list[i] = token
                output_label.append(0)

        return sample_pwy_id_list, output_label, true_label, sample_name_id_list, disease_label_list

