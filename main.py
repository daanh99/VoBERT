import argparse
from pathlib import Path

from bert_pytorch.dataset import WordVocab
from bert_pytorch.predict_log import Predictor
from bert_pytorch.train_log import Trainer
from logdeep.tools.utils import *

from BGL.matrix_options import matrix_options as matrix_options_BGL
from BGL.semantic_options import semantic_options as semantic_options_BGL
from BGL.heuristic_options import heuristic_options as heuristic_options_BGL
from BGL import data_process as data_process_BGL

from HDFS.matrix_options import matrix_options as matrix_options_HDFS
from HDFS.semantic_options import semantic_options as semantic_options_HDFS
from HDFS.heuristic_options import heuristic_options as heuristic_options_HDFS
from HDFS import data_process as data_process_HDFS

from TBird.matrix_options import matrix_options as matrix_options_TBird
from TBird.semantic_options import semantic_options as semantic_options_TBird
from TBird.heuristic_options import heuristic_options as heuristic_options_TBird
from TBird import data_process as data_process_TBird

from private.matrix_options import matrix_options as matrix_options_private
from private.semantic_options import semantic_options as semantic_options_private
from private.heuristic_options import heuristic_options as heuristic_options_private
from private import data_process as data_process_private

seed_everything(seed=1234)


def run(options):
    mode = "semantic" if options['semantic_embeddings'] else "heuristic" if options['unseen_heuristic'] else "matrix"
    print("Running Experiments | " + mode)

    if options["dataset"] == "BGL":
        data_process = data_process_BGL
    elif options["dataset"] == "HDFS":
        data_process = data_process_HDFS
    elif options["dataset"] == "TBird":
        data_process = data_process_TBird
    elif options["dataset"] == "private":
        data_process = data_process_private
    else:
        print("Dataset not supported: ", options["dataset"])
        return

    Path(options['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(options['output_dir'] + "/results/").mkdir(parents=True, exist_ok=True)

    ratio_unseens = find_ratios(options)
    if len(ratio_unseens) == 0:
        print("Generating data...")
        with open(options['output_dir'] + "results/" + "Element Level" + ".csv", "w") as f:
            f.write("ratio,thr,TP,TN,FP,FN,P,R,F1,MCC,AUROC,AUPRC,samples\n")
        with open(options['output_dir'] + "results/" + "Element Level ASO" + ".csv", "w") as f:
            f.write("ratio,thr,TP,TN,FP,FN,P,R,F1,MCC,AUROC,AUPRC,samples\n")
        with open(options['output_dir'] + "results/" + "Sequence Level" + ".csv", "w") as f:
            f.write("ratio,thr,TP,TN,FP,FN,P,R,F1,MCC,AUROC,AUPRC,samples\n")
        with open(options['output_dir'] + "ratio_unseen.csv", "w") as f:
            f.write(
                "ratio_unseen_normal,ratio_unseen_abnormal,ratio_unseen_total,train_size,test_normal_size,test_abnormal_size,train_vocab,test_normal_vocab,test_abnormal_vocab,total_vocab,ratio_unseen_normal_sequences,ratio_unseen_abnormal_sequences,ratio_unseen_total_sequences,mean_seq_length\n")
        data_process.run(options)
    else:
        print("Skipping generating data, it was already found")

    ratio_unseens = find_ratios(options)

    for ratio_unseen in ratio_unseens:
        print("*" * 200)
        print("Running " + mode + " for unseen ratio: " + ratio_unseen)
        print("*" * 200)

        # Set correct options
        options['ratio_unseen'] = ratio_unseen

        # Create output directories if they don't exist
        Path(options['output_dir'] + ratio_unseen).mkdir(parents=True, exist_ok=True)
        Path(options['output_dir'] + ratio_unseen + "/bert/").mkdir(parents=True, exist_ok=True)

        training_done, predicting_done = check_progress(options)

        if not training_done:
            # Create vocab
            with open(options["output_dir"] + "train" + ratio_unseen, 'r') as f:
                seqs = f.readlines()
                elements = []
                for seq in seqs:
                    elements.append([element.split("!;!")[0] for element in seq.split("!|!")[:-1]])
                vocab = WordVocab(elements)

                print("vocab_size", len(vocab))
                vocab.save_vocab(options['output_dir'] + ratio_unseen + "/" + "vocab.pkl")

            Trainer(options).train()
        else:
            print("Skipping training, it was already done")

        if not predicting_done:
            Predictor(options).predict()
        else:
            print("Skipping predicting, it was already done")


def check_progress(options):
    training_done = False
    predicting_done = False
    if os.path.exists(options['output_dir'] + options['ratio_unseen'] + "/bert/best_center_final.pt"):
        training_done = True

    # For heuristic mode
    if options['unseen_heuristic']:
        if os.path.exists(options['output_dir'] + options['ratio_unseen'] + "/vocab.pkl"):
            training_done = True
    try:
        df = pd.read_csv(options['output_dir'] + "results/" + "Sequence Level" + ".csv", dtype={'ratio': str})
        if len(df[df['ratio'] == options['ratio_unseen'][1:]]) >= 2:
            predicting_done = True
    except FileNotFoundError:
        pass

    return training_done, predicting_done


def find_ratios(options):
    # find all files in the output directory that start with test_normal
    ratio_unseens = []
    for file in os.listdir(options["output_dir"]):
        if file.startswith("test_normal"):
            ratio_unseen = "_" + file.split("_")[2]
            ratio_unseens.append(ratio_unseen)
    return ratio_unseens


def run_BGL():
    print("#" * 200)
    print("Running BGL Experiments")
    print("#" * 200)
    run(matrix_options_BGL())
    run(semantic_options_BGL())
    run(heuristic_options_BGL())


def run_TBird():
    print("#" * 200)
    print("Running TBird Experiments")
    print("#" * 200)
    run(matrix_options_TBird())
    run(semantic_options_TBird())
    run(heuristic_options_TBird())


def run_HDFS():
    print("#" * 200)
    print("Running HDFS Experiments")
    print("#" * 200)
    run(matrix_options_HDFS())
    run(semantic_options_HDFS())
    run(heuristic_options_HDFS())


def run_private():
    print("#" * 200)
    print("Running private data Experiments")
    print("#" * 200)
    run(matrix_options_private())
    run(semantic_options_private())
    run(heuristic_options_private())


if __name__ == '__main__':
    print("#" * 200)
    print("Running Experiments")
    print("#" * 200)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs="?", default="all",
                        help='The dataset to run the experiment on: "BGL", "TBird", "HDFS", "private", or "all"')
    args = parser.parse_args()

    if args.dataset == "BGL":
        run_BGL()
    elif args.dataset == "TBird":
        run_TBird()
    elif args.dataset == "HDFS":
        run_HDFS()
    elif args.dataset == "private":
        run_private()
    elif args.dataset == "all":
        print("Running all Experiments (Except private data)")
        run_BGL()
        run_TBird()
        run_HDFS()
    else:
        print("Dataset not supported (new): ", args.dataset)
