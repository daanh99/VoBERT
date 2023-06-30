import pandas as pd


def mean_performance(output_dir):
    df_seq = pd.read_csv(output_dir + "results/Sequence Level.csv").iloc[1::2]
    df_seq["ratio"] = df_seq["ratio"].apply(lambda x: round(x * 100, 2))

    mean_MCC = df_seq["MCC"].mean()
    mean_F1 = df_seq["F1"].mean()
    mean_AUPRC = df_seq["AUROC"].mean()

    print()
    print(output_dir)
    print("Mean MCC: " + str(mean_MCC))
    print("Mean F1: " + str(mean_F1))
    print("Mean AUPRC: " + str(mean_AUPRC))

if __name__ == "__main__":
    # mean_performance("../output/bgl/matrix/")
    mean_performance("../output/bgl/semantic/")
    mean_performance("../output_normal/bgl/semantic/")
    # mean_performance("../output/bgl/heuristic/")

    # mean_performance("../output/tbird/matrix/")
    mean_performance("../output/tbird/semantic/")
    mean_performance("../output_normal/tbird/semantic/")
    # mean_performance("../output/tbird/heuristic/")

    # mean_performance("../output/hdfs/matrix/")
    # mean_performance("../output/hdfs/semantic/")
    # mean_performance("../output/hdfs/heuristic/")
