from torch.utils.data import Dataset
import torch
import random
import numpy as np
from collections import defaultdict

class LogDatasetSemantic(Dataset):
    def __init__(self, log_corpus, time_corpus, vocab, seq_len, embeddings=None, corpus_lines=None, encoding="utf-8", on_memory=True, predict_mode=False, mask_ratio=0.15, per_element_masking=False):
        """

        :param corpus: log sessions/line
        :param vocab: log events collection including pad, ukn ...
        :param seq_len: max sequence length
        :param corpus_lines: number of log sessions
        :param encoding:
        :param on_memory:
        :param predict_mode: if predict
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.encoding = encoding

        self.predict_mode = predict_mode
        self.log_corpus = log_corpus
        self.time_corpus = time_corpus
        self.corpus_lines = len(log_corpus)

        self.mask_ratio = mask_ratio
        self.mask_token = "<mask>"
        self.padding_token = 0
        self.sos_token = "<sos>"

        self.mask_ratio = mask_ratio

        self.embeddings = embeddings
        self.per_element_masking = per_element_masking

        if self.per_element_masking:
            self.key_index = {}
            count = 0
            for log_idx in range(self.corpus_lines):
                k_len = len(self.log_corpus[log_idx])
                mask_index_count = 0
                for mask_index in range(count, count + k_len):
                    self.key_index[mask_index] = [log_idx, mask_index_count]
                    mask_index_count += 1
                    count += 1

    def __len__(self):
        return len(self.key_index) if self.per_element_masking else self.corpus_lines

    def __getitem__(self, idx):
        if self.per_element_masking:
            k, t = self.log_corpus[self.key_index[idx][0]], self.time_corpus[self.key_index[idx][0]]
            k_masked, k_label, t_masked, t_label = self.random_item(k, t, self.key_index[idx][1])
        else:
            k, t = self.log_corpus[idx], self.time_corpus[idx]
            k_masked, k_label, t_masked, t_label = self.random_item(k, t)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        k = [self.sos_token] + k_masked
        k_label = [self.padding_token] + k_label
        # k_label = [self.vocab.sos_index] + k_label

        t = [0] + t_masked
        t_label = [0] + t_label

        return k, k_label, t, t_label

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def random_item(self, k, t, mask_index=None):
        tokens = list(k)
        output_label = []

        time_intervals = list(t)
        time_label = []

        for i, token in enumerate(tokens):
            time_int = time_intervals[i]
            prob = random.random()

            # If predicting, mask token number mask_index
            if self.predict_mode and mask_index is not None:
                if i == mask_index:
                    tokens[i] = self.mask_token
                    output_label.append(token)

                    time_label.append(time_int)
                    time_intervals[i] = 0
                else:
                    tokens[i] = token
                    output_label.append(self.padding_token)
                    time_label.append(0)
                continue

            # replace 15% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                # raise AttributeError("no mask in visualization")

                if self.predict_mode:
                    tokens[i] = self.mask_token
                    output_label.append(token)

                    time_label.append(time_int)
                    time_intervals[i] = 0
                    continue

                prob /= self.mask_ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.mask_token

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = self.vocab.itos[random.randrange(len(self.vocab))]

                # 10% randomly change token to current token
                else:
                    tokens[i] = token

                output_label.append(token)

                time_intervals[i] = 0  # time mask value = 0
                time_label.append(time_int)
            else:
                tokens[i] = token
                output_label.append(self.padding_token)
                time_label.append(0)

        return tokens, output_label, time_intervals, time_label

    def collate_fn(self, batch, percentile=100, dynamical_pad=True):
        lens = [len(seq[0]) for seq in batch]

        # find the max len in each batch
        if dynamical_pad:
            # dynamical padding
            seq_len = int(np.percentile(lens, percentile))
            if self.seq_len is not None:
                seq_len = min(seq_len, self.seq_len)
        else:
            # fixed length padding
            seq_len = self.seq_len

        output = defaultdict(list)
        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]
            time_input = seq[2][:seq_len]
            time_label = seq[3][:seq_len]

            padding = [self.padding_token for _ in range(seq_len - len(bert_input))]
            padding_time = [0 for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), time_input.extend(padding_time), time_label.extend(
                padding_time)

            time_input = np.array(time_input)[:, np.newaxis]

            output["bert_input"].append(torch.tensor(np.array(self.embeddings.embed_sequence(bert_input)), dtype=torch.float))
            output["bert_label"].append(torch.tensor(np.array(self.embeddings.embed_sequence(bert_label)), dtype=torch.float))
            output["time_input"].append(time_input)
            output["time_label"].append(time_label)

        output["bert_input"] = torch.stack(output["bert_input"])
        output["bert_label"] = torch.stack(output["bert_label"])
        output["time_input"] = torch.tensor(np.array(output["time_input"]), dtype=torch.float)
        output["time_label"] = torch.tensor(output["time_label"], dtype=torch.float)

        return output
