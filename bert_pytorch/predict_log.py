import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import torch
from sklearn import metrics
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from bert_pytorch.dataset import WordVocab
from bert_pytorch.dataset.log_dataset import LogDataset
from bert_pytorch.dataset.embeddings import Embeddings
from bert_pytorch.dataset.log_dataset_semantic import LogDatasetSemantic
from bert_pytorch.dataset.sample import fixed_window
from bert_pytorch.dataset.vocab import Vocab
from bert_pytorch.dataset.semantic_vectorizer import bert_classification_embedding


def split_elements_on_label(total_results, total_labels):
    # print(len(total_labels))
    total_labels = np.array(total_labels)
    total_results = np.array(total_results)
    # Separate normal and abnormal labels
    normal_element_index = np.asarray(total_labels == '0').nonzero()
    abnormal_element_index = np.asarray(total_labels == '1').nonzero()

    normal_element_results = total_results[normal_element_index]
    abnormal_element_results = total_results[abnormal_element_index]

    # print("normal element results " + str(len(normal_element_results)))
    # print("abnormal element results " + str(len(abnormal_element_results)))

    return normal_element_results, abnormal_element_results


class Predictor():
    def __init__(self, options):
        self.vocab = None
        self.evaluate_element_level = options["evaluate_element_level"]
        self.per_element_masking = options["per_element_masking"]
        self.semantic_embeddings = options['semantic_embeddings']
        self.threshold_on_valid = options["threshold_on_valid"]

        self.model_dir = options["output_dir"] + options["ratio_unseen"] + "/bert/"
        self.model_path = self.model_dir + "best_bert.pth"
        self.vocab_path = options["output_dir"] + options["ratio_unseen"] + "/vocab.pkl"
        self.device = options["device"]
        self.window_size = options["window_size"]
        self.adaptive_window = options["adaptive_window"]
        self.seq_len = options["seq_len"]
        self.corpus_lines = options["corpus_lines"]
        self.on_memory = options["on_memory"]
        self.batch_size = 20 if options["semantic_embeddings"] else 70
        self.num_workers = options["num_workers"]
        self.num_candidates = options["num_candidates"]
        self.output_dir = options["output_dir"]
        self.gaussian_mean = options["gaussian_mean"]
        self.gaussian_std = options["gaussian_std"]
        self.hidden = options["hidden"]

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.scale_path = options["scale_path"]

        self.hypersphere_loss = options["hypersphere_loss"]

        self.hypersphere_loss_test = options["hypersphere_loss_test"]

        self.lower_bound = self.gaussian_mean - 3 * self.gaussian_std
        self.upper_bound = self.gaussian_mean + 3 * self.gaussian_std

        self.center = None
        self.radius = None
        self.test_ratio = options["test_ratio"]
        self.mask_ratio = options["mask_ratio"]
        self.min_len = options["min_len"]

        self.thr_seq = None
        self.thr_el = None
        self.thr_el_aso = None

        self.ratio_unseen = options["ratio_unseen"]
        self.dataset = options["dataset"]
        self.unseen_heuristic = options["unseen_heuristic"]

    def detect_logkey_anomaly(self, masked_output, masked_label):
        num_undetected_tokens = 0
        output_masks = []
        token_indices = []
        if not self.semantic_embeddings:
            for i, token in enumerate(masked_label):
                if token not in torch.argsort(-masked_output[i])[:self.num_candidates]:
                    num_undetected_tokens += 1
                token_indices.append(
                    ((torch.argsort(-masked_output[i]) == token).nonzero(as_tuple=True)[0][0].cpu().numpy()))

        if self.semantic_embeddings:
            # token_indices_mean = nn.CosineEmbeddingLoss()(masked_output, masked_label, target=torch.ones([masked_output.shape[0]]).to(self.device)).detach().cpu().numpy()
            token_indices_mean = nn.MSELoss()(masked_output, masked_label).detach().cpu().numpy()
        else:
            token_indices_mean = np.mean(token_indices)

        return num_undetected_tokens, [output_masks, masked_label.cpu().numpy()], token_indices_mean

    def generate_test(self, output_dir, file_name, window_size, adaptive_window, seq_len, scale, min_len):
        """
        :return: log_seqs: num_samples x session(seq)_length, tim_seqs: num_samples x session_length
        """
        log_seqs = []
        tim_seqs = []

        if file_name.endswith(".pkl"):
            with open(output_dir + file_name, "rb") as f:
                log_seqs = pickle.load(f)
                # dummy value for time
                if self.dataset != "HDFS":
                    tim_seqs = [[[0, 0] for _ in range(len(seq))] for seq in log_seqs]
                else:
                    tim_seqs = [[0 for _ in range(len(seq))] for seq in log_seqs]
        else:
            with open(output_dir + file_name, "r") as f:
                # compute num lines and go back to first line
                num_lines = len(f.readlines())
                f.seek(0)
                for idx, line in tqdm(enumerate(f.readlines()), total=num_lines,
                                      desc="generating test from file: " + file_name):
                    # if idx > 40: break
                    log_seq, tim_seq = fixed_window(line, window_size,
                                                    adaptive_window=adaptive_window,
                                                    seq_len=seq_len, min_len=min_len)
                    if len(log_seq) == 0:
                        continue
                    log_seqs += log_seq
                    tim_seqs += tim_seq

        # sort seq_pairs by seq len
        log_seqs = np.array(log_seqs, dtype=object)
        tim_seqs = np.array(tim_seqs, dtype=object)

        test_len = list(map(len, log_seqs))
        test_sort_index = np.argsort(-1 * np.array(test_len))

        log_seqs = log_seqs[test_sort_index]
        tim_seqs = tim_seqs[test_sort_index]

        print(f"{file_name} size: {len(log_seqs)}")
        return log_seqs, tim_seqs

    def helper(self, model, output_dir, file_name, vocab, saving_file_prefix, scale=None, error_dict=None):
        logkey_test, time_test = self.generate_test(output_dir, file_name, self.window_size, self.adaptive_window,
                                                    self.seq_len, scale, self.min_len)
        if self.test_ratio != 1:
            num_test = len(logkey_test)
            rand_index = torch.randperm(num_test)
            rand_index = rand_index[:int(num_test * self.test_ratio)] if isinstance(self.test_ratio,
                                                                                    float) else rand_index[
                                                                                                :self.test_ratio]
            logkey_test, time_test = logkey_test[rand_index], time_test[rand_index]

        print(f"test data size after applying test_ratio: {len(logkey_test)}")

        logkey_test = self.remove_last_element_of_max_len_seq(logkey_test)
        time_test = self.remove_last_element_of_max_len_seq(time_test)

        if self.unseen_heuristic:
            if self.dataset != 'HDFS':
                # Remove element level label data
                logkey_test = [[logkey[0] for logkey in seq] for seq in logkey_test]

            total_results = self.unseen_heuristic_predict(logkey_test)
            return total_results, None

        if self.dataset != "HDFS":
            # Element level label data
            logkey_test_element_labels = [[logkey[1] for logkey in seq] for seq in logkey_test]

            # Remove element level label data
            logkey_test = [[logkey[0] for logkey in seq] for seq in logkey_test]
            time_test = [[time[0] for time in seq] for seq in time_test]
        else:
            logkey_test_element_labels = []

        sequence_lengths = [len(logkey_test[seq_i]) for seq_i in range(len(logkey_test))]

        if self.per_element_masking:
            print("Saving normal sequence lenghts")
            with open(self.model_dir + saving_file_prefix + "_sequence_lengths.pkl", "wb") as f:
                pickle.dump(sequence_lengths, f)

        if self.evaluate_element_level:
            print("Saving test normal element labels")
            with open(self.model_dir + saving_file_prefix + "_element_labels.pkl", "wb") as f:
                pickle.dump(logkey_test_element_labels, f)

        if self.semantic_embeddings:
            print("Extending vocab with new logkeys")
            print(vocab)
            vocab = self.extend_vocab(vocab, logkey_test)
            vocab.vocab_rerank()
            print(vocab)

            embeddings = Embeddings(
                {x[0]: x[1] for x in
                 zip(vocab.stoi.keys(), bert_classification_embedding(vocab.stoi.keys(), self.device))}, self.hidden)

            seq_dataset = LogDatasetSemantic(logkey_test, time_test, vocab, seq_len=self.seq_len, embeddings=embeddings,
                                             corpus_lines=self.corpus_lines, on_memory=self.on_memory,
                                             predict_mode=True,
                                             mask_ratio=self.mask_ratio, per_element_masking=self.per_element_masking)
        else:
            seq_dataset = LogDataset(logkey_test, time_test, vocab, seq_len=self.seq_len,
                                     corpus_lines=self.corpus_lines, on_memory=self.on_memory, predict_mode=True,
                                     mask_ratio=self.mask_ratio, per_element_masking=self.per_element_masking)

        print(f"Dataset size after per element masking: {len(seq_dataset)}")
        print(f"Dataset size after per element masking without optimization: {len(seq_dataset) * self.seq_len}")
        print(f"Element label amount of sequences: {len(logkey_test_element_labels)}")
        print(f"Element label size flat: {len([item for sublist in logkey_test_element_labels for item in sublist])}")

        del sequence_lengths, logkey_test_element_labels

        total_results, output_debug = self.run_model(seq_dataset, model, file_name)

        return total_results, output_debug

    def run_model(self, seq_dataset, model, file_name):
        total_results = []
        output_results = []
        total_dist = []
        output_cls = []

        data_loader = DataLoader(seq_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                 collate_fn=seq_dataset.collate_fn)

        skip_counter = 0
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Predicting " + file_name):
            data = {key: value.to(self.device) for key, value in data.items()}

            result = model(data["bert_input"], data["time_input"])

            mask_lm_output, mask_tm_output = result["logkey_output"], result["time_output"]
            output_cls += result["cls_output"].tolist()
            # loop though each log sequence in batch
            for i in range(len(data["bert_label"])):
                seq_results = {"num_error": 0,
                               "undetected_tokens": 0,
                               "masked_tokens": 0,
                               "total_logkey": 0,
                               "deepSVDD_label": 0,
                               "element_anomaly_score": 0
                               }

                if self.semantic_embeddings:
                    mask_index = data["bert_label"][i].sum(1) != 0
                    num_masked = torch.sum(mask_index).tolist()
                    seq_results["masked_tokens"] = num_masked

                    seq_results["total_logkey"] = torch.sum(
                        data["bert_input"][i].sum(1) != 0).item()
                else:
                    mask_index = data["bert_label"][i] > 0
                    num_masked = torch.sum(mask_index).tolist()
                    seq_results["masked_tokens"] = num_masked

                    seq_results["total_logkey"] = torch.sum(
                        data["bert_input"][i] > 0).item()

                if self.per_element_masking:
                    assert seq_results[
                               "masked_tokens"] == 1, "There should always be 1 element masked, but # masked was: " + str(
                        seq_results["masked_tokens"])
                else:
                    if seq_results["masked_tokens"] == 0:
                        skip_counter += 1
                        continue  # skip if no masked tokens

                if self.is_logkey:
                    num_undetected, output_seq, token_indices = self.detect_logkey_anomaly(
                        mask_lm_output[i][mask_index], data["bert_label"][i][mask_index])
                    seq_results["undetected_tokens"] = num_undetected
                    seq_results["element_anomaly_score"] = token_indices
                    # output_results.append(output_seq)

                if self.hypersphere_loss_test:
                    # detect by deepSVDD distance
                    assert result["cls_output"][i].size() == self.center.size()
                    # dist = torch.sum((result["cls_fnn_output"][i] - self.center) ** 2)
                    dist = torch.sqrt(torch.sum((result["cls_output"][i] - self.center) ** 2))
                    total_dist.append(dist.item())

                    # user defined threshold for deepSVDD_label
                    seq_results["deepSVDD_label"] = int(dist.item() > self.radius)
                # print(seq_results["element_anomaly_score"])
                total_results.append(seq_results)
        print("Skipped sequences because zero masked elements: " + str(skip_counter))

        return total_results, output_results

    def main_predict_function(self, thr_on_validation=False):
        start_time = time.time()
        model = None
        scale = None
        error_dict = None

        self.vocab = WordVocab.load_vocab(self.vocab_path)
        print("vocab length: {}".format(len(self.vocab)))

        if not self.unseen_heuristic:
            model = torch.load(self.model_path)
            model.to(self.device)
            model.eval()
            print('model_path: {}'.format(self.model_path))

            if self.is_time:
                with open(self.scale_path, "rb") as f:
                    scale = pickle.load(f)

                with open(self.model_dir + "error_dict.pkl", 'rb') as f:
                    error_dict = pickle.load(f)

            if self.hypersphere_loss:
                center_dict = torch.load(self.model_dir + "best_center.pt")
                self.center = center_dict["center"]
                self.radius = center_dict["radius"]
                # self.center = self.center.view(1,-1)

        print("test normal predicting")
        test_normal_results, test_normal_errors = self.helper(
            model, self.output_dir,
            "test_normal" + self.ratio_unseen if not thr_on_validation else "valid_normal" + self.ratio_unseen + ".pkl",
            self.vocab, "test_normal", scale,
            error_dict)

        print("Saving test normal results")
        with open(self.model_dir + "test_normal_results.pkl", "wb") as f:
            pickle.dump(test_normal_results, f)

        print("Saving test normal errors")
        with open(self.model_dir + "test_normal_errors.pkl", "wb") as f:
            pickle.dump(test_normal_errors, f)

        del test_normal_results, test_normal_errors

        print("test abnormal predicting")
        test_abnormal_results, test_abnormal_errors = self.helper(
            model, self.output_dir,
            "test_abnormal" if not thr_on_validation else "valid_abnormal", self.vocab,
            "test_abnormal",
            scale,
            error_dict)

        print("Saving test abnormal results")
        with open(self.model_dir + "test_abnormal_errors.pkl", "wb") as f:
            pickle.dump(test_abnormal_errors, f)

        print("Saving test abnormal results")
        with open(self.model_dir + "test_abnormal_results.pkl", "wb") as f:
            pickle.dump(test_abnormal_results, f)

        elapsed_time = time.time() - start_time
        print('elapsed_time (mins): {}'.format(elapsed_time / 60))

    def predict(self):
        print("#" * 100)
        print("Predicting. Semantic mode: " + str(self.semantic_embeddings) + " Per element masking: " + str(
            self.per_element_masking) + " Using Valid set to find thr: " + str(
            self.threshold_on_valid) + " unseen ratio: " + str(self.ratio_unseen))
        print("#" * 100)

        if self.threshold_on_valid:
            self.main_predict_function(thr_on_validation=True)
            thr_seq, thr_el, thr_el_aso = self.analyse_predictions()
            self.thr_seq = thr_seq
            self.thr_el = thr_el
            self.thr_el_aso = thr_el_aso

        self.main_predict_function(thr_on_validation=False)
        self.analyse_predictions()

    def analyse_predictions(self):
        print("Loading test results from pickle files")
        test_normal_results = pickle.load(open(self.model_dir + "test_normal_results.pkl", "rb"))
        test_abnormal_results = pickle.load(open(self.model_dir + "test_abnormal_results.pkl", "rb"))
        # print(test_normal_results)
        # print(test_abnormal_results)

        if self.per_element_masking:
            normal_sequence_lengths = pickle.load(open(self.model_dir + "test_normal_sequence_lengths.pkl", "rb"))
            abnormal_sequence_lengths = pickle.load(open(self.model_dir + "test_abnormal_sequence_lengths.pkl", "rb"))

            print("Sequences " + str(len(normal_sequence_lengths) + len(abnormal_sequence_lengths)))

            normal_sequence_results = aggregate_elements_back_into_sequence(normal_sequence_lengths,
                                                                            test_normal_results)
            abnormal_sequence_results = aggregate_elements_back_into_sequence(abnormal_sequence_lengths,
                                                                              test_abnormal_results)
        else:
            normal_sequence_results = test_normal_results
            abnormal_sequence_results = test_abnormal_results

        if self.evaluate_element_level:
            test_normal_element_labels = pickle.load(open(self.model_dir + "test_normal_element_labels.pkl", "rb"))
            test_abnormal_element_labels = pickle.load(open(self.model_dir + "test_abnormal_element_labels.pkl", "rb"))

            # flatten lists
            test_normal_element_labels = [item for sublist in test_normal_element_labels for item in sublist]
            test_abnormal_element_labels = [item for sublist in test_abnormal_element_labels for item in sublist]

            print("test_normal_element_labels: " + str(len(test_normal_element_labels)))
            print("test_normal_results: " + str(len(test_normal_results)))

            print("test_abnormal_element_labels: " + str(len(test_abnormal_element_labels)))
            print("test_abnormal_results: " + str(len(test_abnormal_results)))

            assert len(test_normal_results) == len(test_normal_element_labels)
            assert len(test_abnormal_results) == len(test_abnormal_element_labels)

            total_results = test_normal_results + test_abnormal_results
            total_labels = test_normal_element_labels + test_abnormal_element_labels

            normal_element_results, abnormal_element_results = split_elements_on_label(total_results, total_labels)
            normal_element_results_abnormal_seq_only, abnormal_element_results_abnormal_seq_only = \
                split_elements_on_label(
                    test_abnormal_results, test_abnormal_element_labels)

            th_range_element = np.arange(0, 2, 0.001) if self.semantic_embeddings else np.arange(0, min(len(self.vocab),
                                                                                                        200), 1)

            best_thr_element = self.print_results(normal_element_results,
                                                  abnormal_element_results,
                                                  "Element Level",
                                                  'element_anomaly_score',
                                                  [self.thr_el] if self.thr_el is not None else th_range_element)
            best_thr_element_aso = self.print_results(normal_element_results_abnormal_seq_only,
                                                      abnormal_element_results_abnormal_seq_only,
                                                      "Element Level ASO",
                                                      'element_anomaly_score',
                                                      [self.thr_el_aso] if self.thr_el_aso is not None
                                                      else th_range_element)
        th_range_sequence = np.arange(0, 1, 0.001) if self.unseen_heuristic else np.arange(0, 1, 0.001)
        best_thr_sequence = self.print_results(normal_sequence_results,
                                               abnormal_sequence_results,
                                               "Sequence Level",
                                               'element_anomaly_score' if (
                                                       self.semantic_embeddings or self.unseen_heuristic) else 'ratio_undetected',
                                               [self.thr_seq] if self.thr_seq is not None else th_range_sequence)
        return best_thr_sequence, best_thr_element if self.evaluate_element_level else -1, best_thr_element_aso if self.evaluate_element_level else -1

    def print_results(self, normal_results, abnormal_results, name, sequence_score, th_range):
        params = {"is_logkey": self.is_logkey, "is_time": self.is_time, "hypersphere_loss": self.hypersphere_loss,
                  "hypersphere_loss_test": self.hypersphere_loss_test}

        print("normal_results: " + str(len(normal_results)))
        print("abnormal_results: " + str(len(abnormal_results)))

        thr, FP, TP, TN, FN, P, R, F1, MCC = find_best_threshold(normal_results,
                                                                 abnormal_results,
                                                                 params=params,
                                                                 thr_range=th_range,
                                                                 sequence_score=sequence_score,
                                                                 name=name)
        auroc = 0
        auprc = 0
        if len(normal_results) > 0 and len(abnormal_results) > 0:
            auroc, auprc = self.calculate_auc(normal_results, abnormal_results, name=name,
                                     sequence_score=sequence_score)
        samples = len(normal_results) + len(abnormal_results)
        output = '''
            ------------- {} Results -------------
            Samples: {}
            Best threshold: {}
            TP: {}, TN: {}, FP: {}, FN: {}
            Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%, MCC: {:.2f}, AUROC: {:.2f},  AUPRC: {:.2f}'''.format(
            name,
            samples,
            thr,
            TP, TN, FP, FN,
            P, R, F1, MCC, auroc, auprc)

        print(output)

        # write results to csv file
        with open(self.output_dir + "results/" + name + ".csv", "a") as f:
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(self.ratio_unseen[1:], thr, TP, TN, FP, FN, P, R, F1,
                                                                   MCC, auroc, auprc, samples))

        return thr

    def remove_last_element_of_max_len_seq(self, logkey_test):
        if self.seq_len is not None:
            for i in range(len(logkey_test)):
                if len(logkey_test[i]) == self.seq_len:
                    logkey_test[i] = logkey_test[i][:-1]
        return logkey_test

    def extend_vocab(self, vocab, logkey_test):
        freqs = {}
        for logkey in [item for sublist in logkey_test for item in sublist]:
            freqs[logkey] = freqs.get(logkey, 0) + 1
        vocab.extend(Vocab(freqs))
        return vocab

    def calculate_auc(self, test_normal_results, test_abnormal_results, name="logbert",
                      sequence_score="ratio_undetected"):
        if sequence_score == "ratio_undetected":
            normal_scores = [
                seq_res["undetected_tokens"] / seq_res["masked_tokens"] if seq_res["masked_tokens"] != 0 else 0
                for seq_res in test_normal_results]
            abnormal_scores = [
                seq_res["undetected_tokens"] / seq_res["masked_tokens"] if seq_res["masked_tokens"] != 0 else 0 for
                seq_res
                in test_abnormal_results]
        else:
            normal_scores = [seq_res['element_anomaly_score'] for seq_res in test_normal_results]
            abnormal_scores = [seq_res["element_anomaly_score"] for seq_res in test_abnormal_results]

        scores = normal_scores + abnormal_scores
        labels = [0] * len(normal_scores) + [1] * len(abnormal_scores)

        # Calculate ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
        display.plot()
        plt.savefig(self.output_dir + "results/" + name + self.ratio_unseen + 'AUCROC.png')
        plt.close()

        # Calculate precision-recall curve
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=1)
        pr_auc = metrics.auc(recall, precision)
        display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc,
                                                 estimator_name=name)
        display.plot()
        plt.savefig(self.output_dir + "results/" + name + self.ratio_unseen + 'AUPRC.png')
        plt.close()

        return roc_auc, pr_auc

    def unseen_heuristic_predict(self, logkey_test):
        # Predict using the unseen heuristic: Just return the percentage of unseen tokens in the sequence
        # as the anomaly score
        anomaly_score = []
        for seq in logkey_test:
            # check how may logkeys in the sequence are unseen
            ratio_unseen = {'element_anomaly_score': len(
                [key_label_pair for key_label_pair in seq if key_label_pair not in self.vocab.itos]) / len(seq)}
            anomaly_score.append(ratio_unseen)
        return anomaly_score


def aggregate_elements_back_into_sequence(sequence_lengths, elements):
    final_results = []
    count = 0
    for seq_i in range(len(sequence_lengths)):
        seq = {"num_error": 0, "undetected_tokens": 0, "masked_tokens": 0, "total_logkey": 0, 'deepSVDD_label': 0,
               'element_anomaly_score': 0}
        # print("sequence " + str(seq_i) + " length is " + str(sequence_lengths[seq_i]))
        for mask_j in range(count, count + sequence_lengths[seq_i]):
            count += 1
            seq_i_mask_j = elements[mask_j]
            seq["num_error"] += seq_i_mask_j["num_error"]
            seq["undetected_tokens"] += seq_i_mask_j["undetected_tokens"]
            seq["masked_tokens"] += seq_i_mask_j["masked_tokens"]
            seq["total_logkey"] = seq_i_mask_j["total_logkey"]
            seq['deepSVDD_label'] = seq_i_mask_j['deepSVDD_label']
            seq['element_anomaly_score'] += seq_i_mask_j['element_anomaly_score']
        seq["num_error"] = seq["num_error"] / sequence_lengths[seq_i]
        seq["element_anomaly_score"] = seq["element_anomaly_score"] / sequence_lengths[seq_i]
        final_results.append(seq)

    return final_results


def compute_anomaly(results, params, seq_threshold=0.5, sequence_score='ratio_undetected'):
    is_logkey = params["is_logkey"]
    is_time = params["is_time"]
    total_errors = 0
    for seq_res in results:
        # label sequences as anomaly when over half of masked tokens are undetected
        if sequence_score == 'ratio_undetected':
            percentage_undetected = seq_res["undetected_tokens"] / seq_res["masked_tokens"] if seq_res[
                                                                                                   "masked_tokens"] != 0 else 0
            anomaly_score = percentage_undetected
        # if percentage_undetected > 0: print(percentage_undetected)
        else:
            anomaly_score = seq_res["element_anomaly_score"]
            # print("anomaly score " + str(percentage_undetected) + " > " + str(seq_threshold) + " " + str(percentage_undetected > seq_threshold))
        if (is_logkey and anomaly_score >= seq_threshold) or \
                (is_time and anomaly_score >= seq_threshold) or \
                (params["hypersphere_loss_test"] and seq_res["deepSVDD_label"]):
            total_errors += 1
        # if evaluation_level == 'element':
        #     print("anomaly_score: " + str(total_errors))
    return total_errors


def find_best_threshold(test_normal_results, test_abnormal_results, params, thr_range,
                        sequence_score='ratio_undetected', name=""):
    best_result = [-1000] * 9

    # print(test_normal_results)
    # print(test_abnormal_results)
    # print(str(seq_range))
    for thr in tqdm(thr_range, total=len(thr_range), desc="Finding best threshold for " + name):
        FP = float(compute_anomaly(test_normal_results, params, thr, sequence_score))
        TP = float(compute_anomaly(test_abnormal_results, params, thr, sequence_score))

        TN = float(len(test_normal_results) - FP)
        FN = float(len(test_abnormal_results) - TP)
        P = 100 * TP / (TP + FP) if (TP + FP) != 0 else 0
        R = 100 * TP / (TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0
        MCC = 100 * (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (
                (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0) else 0

        # print("trying " + str(seq_th) + " " + str(F1))

        if MCC > best_result[-1]:
            best_result = [thr, FP, TP, TN, FN, P, R, F1, MCC]
    return best_result
