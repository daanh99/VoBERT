import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_train_valid_loss(model_dir, surfix_log, mode):
    train_loss = pd.read_csv(model_dir + f"train{surfix_log}.csv")
    valid_loss = pd.read_csv(model_dir + f"valid{surfix_log}.csv")
    sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
    sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")
    plt.title(mode + " | epoch loss curve")
    plt.legend()
    plt.savefig(model_dir + "train_valid_loss_partial.png")
    plt.show()
    plt.close()


def plot_ratio_unseen_x_axis(output_dir):
    df_ratio_unseen = pd.read_csv(output_dir + f"ratio_unseen.csv")

    sns.lineplot(x="ratio_unseen_normal", y="train_size", data=df_ratio_unseen, label="train")
    sns.lineplot(x="ratio_unseen_normal", y="test_normal_size", data=df_ratio_unseen, label="test normal")
    sns.lineplot(x="ratio_unseen_normal", y="test_abnormal_size", data=df_ratio_unseen, label="test abnormal")
    plt.title("Dataset size")
    plt.ylabel("Size")
    plt.legend()
    plt.savefig(output_dir + "ratio_unseen_dataset.png")
    plt.show()
    plt.close()

    sns.lineplot(x="ratio_unseen_normal", y="train_vocab", data=df_ratio_unseen, label="train")
    sns.lineplot(x="ratio_unseen_normal", y="test_normal_vocab", data=df_ratio_unseen, label="test normal")
    sns.lineplot(x="ratio_unseen_normal", y="test_abnormal_vocab", data=df_ratio_unseen, label="test abnormal")
    plt.title("Vocab size")
    plt.ylabel("Size")
    plt.legend()
    plt.savefig(output_dir + "ratio_unseen_vocab.png")
    plt.show()
    plt.close()

    sns.lineplot(x="ratio_unseen_normal", y="ratio_unseen_normal_sequences", data=df_ratio_unseen, label="normal")
    sns.lineplot(x="ratio_unseen_normal", y="ratio_unseen_abnormal_sequences", data=df_ratio_unseen, label="abnormal")
    sns.lineplot(x="ratio_unseen_normal", y="ratio_unseen_total_sequences", data=df_ratio_unseen, label="total")
    plt.title("Percentage of sequences containing unseen")
    plt.ylabel("% unseen")
    plt.legend()
    plt.savefig(output_dir + "ratio_unseen_seqs.png")
    plt.show()
    plt.close()


def plot_performance_vs_unseen(output_dir, dataset_name, output_dir_corrected):
    df_seq_matrix_corrected = pd.read_csv(output_dir_corrected + "matrix/ratio_unseen.csv").reset_index(drop=True)

    df_seq_matrix = pd.read_csv(output_dir + "matrix/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_seq_matrix["ratio"] = df_seq_matrix_corrected["ratio_unseen_normal_sequences"].apply(lambda x: round(x * 100, 2))

    df_seq_semantic = pd.read_csv(output_dir + "semantic/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_seq_semantic["ratio"] = df_seq_matrix_corrected["ratio_unseen_normal_sequences"].apply(lambda x: round(x * 100, 2))

    df_seq_heuristic = pd.read_csv(output_dir + "heuristic/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_seq_heuristic["ratio"] = df_seq_matrix_corrected["ratio_unseen_normal_sequences"].apply(lambda x: round(x * 100, 2))

    print(df_seq_matrix_corrected["ratio_unseen_normal_sequences"])
    print(df_seq_matrix["ratio"])


    sns.lineplot(x="ratio", y="MCC", data=df_seq_semantic, label="VoBERT", markers=True, marker="o")
    sns.lineplot(x="ratio", y="MCC", data=df_seq_matrix, label="LogBERT", markers=True, marker="o")
    sns.lineplot(x="ratio", y="MCC", data=df_seq_heuristic, label="Unseen Logkey Heuristic", markers=True, marker="o")
    # plt.xticks(df_seq_semantic["ratio"].apply(lambda x: round(x,0)))
    # print(df_seq_semantic["ratio"].apply(lambda x: round(x,0)))
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    plt.title("Performance on Sequence Level | " + dataset_name)
    plt.ylabel("MCC Score")
    plt.xlabel("% of normal sequences containing at least one unseen element")
    plt.legend()
    plt.savefig(output_dir + "performance_sequence_" + dataset_name + ".png")
    plt.show()
    plt.close()

    # Bonus plot for unseen ratio of normal sequences
    # df_ratio_unseen = pd.read_csv(output_dir + "matrix/ratio_unseen.csv")
    # df_ratio_unseen["ratio_unseen_normal_sequences"] = df_ratio_unseen["ratio_unseen_normal_sequences"].apply(
    #     lambda x: round(x * 100, 2))
    # df_ratio_unseen["ratio_unseen_normal"] = df_ratio_unseen["ratio_unseen_normal"].apply(lambda x: round(x * 100, 2))
    # replacements = df_seq_matrix['ratio'].map(
    # df_ratio_unseen.set_index('ratio_unseen_normal_sequences')['ratio_unseen_normal'])
    # df_seq_matrix['ratio_unseen_raw'] = replacements
    # df_seq_semantic['ratio_unseen_raw'] = replacements
    # sns.lineplot(x="ratio_unseen_raw", y="MCC", data=df_seq_semantic, label="VF-MLM")
    # sns.lineplot(x="ratio_unseen_raw", y="MCC", data=df_seq_matrix, label="MLM")
    # # plt.xticks(df_seq_matrix["ratio_unseen_raw"])
    # plt.ylim(0, 100)
    # plt.title("Performance on Sequence Level: unseen elements")
    # plt.ylabel("MCC Score")
    # plt.xlabel("% unique elements unseen in the normal test set")
    # plt.legend()
    # plt.savefig(output_dir + "performance_sequence_ratio_unseen.png")
    # plt.show()
    # plt.close()

    # df_el_matrix = pd.read_csv(output_dir + "matrix/results/Element Level.csv").iloc[1::2]
    # df_el_matrix["ratio"] = df_el_matrix["ratio"].apply(lambda x: round(x * 100, 2))
    # df_el_semantic = pd.read_csv(output_dir + "semantic/results/Element Level.csv").iloc[1::2]
    # df_el_semantic["ratio"] = df_el_semantic["ratio"].apply(lambda x: round(x * 100, 2))
    #
    # sns.lineplot(x="ratio", y="MCC", data=df_el_semantic, label="VF-MLM")
    # sns.lineplot(x="ratio", y="MCC", data=df_el_matrix, label="MLM")
    # plt.xticks(df_el_semantic["ratio"])
    # plt.ylim(0, 100)
    # plt.title("Performance on Element Level")
    # plt.ylabel("MCC Score")
    # plt.xlabel("% of normal sequences containing at least one unseen element")
    # plt.legend()
    # plt.savefig(output_dir + "performance_el.png")
    # plt.show()
    # plt.close()
    #
    # df_el_aso_matrix = pd.read_csv(output_dir + "matrix/results/Element Level ASO.csv").iloc[1::2]
    # df_el_aso_matrix["ratio"] = df_el_aso_matrix["ratio"].apply(lambda x: round(x * 100, 2))
    # df_el_aso_semantic = pd.read_csv(output_dir + "semantic/results/Element Level ASO.csv").iloc[1::2]
    # df_el_aso_semantic["ratio"] = df_el_aso_semantic["ratio"].apply(lambda x: round(x * 100, 2))
    #
    # sns.lineplot(x="ratio", y="MCC", data=df_el_aso_semantic, label="VF-MLM")
    # sns.lineplot(x="ratio", y="MCC", data=df_el_aso_matrix, label="MLM")
    # plt.xticks(df_el_aso_semantic["ratio"])
    # plt.ylim(0, 100)
    # plt.title("Performance on Element ASO Level")
    # plt.ylabel("MCC Score")
    # plt.xlabel("% of normal sequences containing at least one unseen element")
    # plt.legend()
    # plt.savefig(output_dir + "performance_el_aso.png")
    # plt.show()
    # plt.close()
    #


def plot_parsing_vs_no_parsing(dataset_name, output_dir_corrected):
    df_seq_matrix_corrected = pd.read_csv(output_dir_corrected + "matrix/ratio_unseen.csv")
    df_seq_semantic_parsing = pd.read_csv(
        "../output_parsing/" + dataset_name + "/semantic/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_seq_semantic_parsing["ratio"] = df_seq_matrix_corrected["ratio_unseen_normal_sequences"].apply(lambda x: round(x * 100, 2))

    df_seq_semantic_no_parsing = pd.read_csv(
        "../output_no_parsing/" + dataset_name + "/semantic/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_seq_semantic_no_parsing["ratio"] = df_seq_matrix_corrected["ratio_unseen_normal_sequences"].apply(lambda x: round(x * 100, 2))

    sns.lineplot(x="ratio", y="MCC", data=df_seq_semantic_parsing, label="VoBERT Parsing", markers=True, marker="o")
    sns.lineplot(x="ratio", y="MCC", data=df_seq_semantic_no_parsing, label="VoBERT No Parsing", markers=True,
                 marker="o")
    # plt.xticks(df_seq_semantic["ratio"].apply(lambda x: round(x,0)))
    # print(df_seq_semantic["ratio"].apply(lambda x: round(x,0)))
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    plt.title("Performance on Sequence Level | " + dataset_name)
    plt.ylabel("MCC Score")
    plt.xlabel("% of normal sequences containing at least one unseen element")
    plt.legend()
    plt.savefig("../plots/performance_sequence_parsing_comp_" + dataset_name + ".png")
    plt.show()
    plt.close()


def plot_el_vs_seq(dataset_name):
    df_seq_semantic = pd.read_csv("output_parsing/" + dataset_name + "/semantic/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_seq_semantic["ratio"] = df_seq_semantic["ratio"].apply(lambda x: round(x * 100, 2))

    df_el_semantic = pd.read_csv("output_el_level/" + dataset_name + "/semantic/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_el_semantic["ratio"] = df_el_semantic["ratio"].apply(lambda x: round(x * 100, 2))

    sns.lineplot(x="ratio", y="MCC", data=df_seq_semantic, label="VF-MLM Ratio Masking", markers=True, marker="o")
    sns.lineplot(x="ratio", y="MCC", data=df_el_semantic, label="VF-MLM Per-element Masking", markers=True, marker="o")
    # plt.xticks(df_seq_semantic["ratio"].apply(lambda x: round(x,0)))
    # print(df_seq_semantic["ratio"].apply(lambda x: round(x,0)))
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    plt.title("Performance on Sequence Level | " + dataset_name)
    plt.ylabel("MCC Score")
    plt.xlabel("% of normal sequences containing at least one unseen element")
    plt.legend()
    plt.savefig("plots/performance_sequence_ratio_per-element_masking_comp_" + dataset_name + ".png")
    plt.show()
    plt.close()


def plot_ratio_unseen(output_dir, name):
    df_ratio_unseen = pd.read_csv(output_dir + f"ratio_unseen.csv")

    # convert ratio's to percentage
    df_ratio_unseen["ratio_unseen_normal"] = df_ratio_unseen["ratio_unseen_normal"].apply(lambda x: round(x * 100, 2))
    df_ratio_unseen["ratio_unseen_abnormal"] = df_ratio_unseen["ratio_unseen_abnormal"].apply(
        lambda x: round(x * 100, 2))
    df_ratio_unseen["ratio_unseen_total"] = df_ratio_unseen["ratio_unseen_total"].apply(lambda x: round(x * 100, 2))
    df_ratio_unseen["ratio_unseen_normal_sequences"] = df_ratio_unseen["ratio_unseen_normal_sequences"].apply(
        lambda x: round(x * 100, 2))
    df_ratio_unseen["ratio_unseen_abnormal_sequences"] = df_ratio_unseen["ratio_unseen_abnormal_sequences"].apply(
        lambda x: round(x * 100, 2))
    df_ratio_unseen["ratio_unseen_total_sequences"] = df_ratio_unseen["ratio_unseen_total_sequences"].apply(
        lambda x: round(x * 100, 2))

    sns.lineplot(x=df_ratio_unseen.index, y="ratio_unseen_normal", data=df_ratio_unseen, label="Normal test sequences", marker="o")
    sns.lineplot(x=df_ratio_unseen.index, y="ratio_unseen_abnormal", data=df_ratio_unseen, label="Abnormal test sequences", marker="o")
    sns.lineplot(x=df_ratio_unseen.index, y="ratio_unseen_total", data=df_ratio_unseen, label="Total test sequences", marker="o")
    plt.xticks(df_ratio_unseen.index)
    plt.title("Ratio unseen in test set | " + name)
    plt.ylabel("% unseen")
    plt.xlabel("Data redistribution algorithm iterations")
    plt.legend()
    plt.savefig(output_dir + "ratio_unseen_ratio.png")
    plt.show()
    plt.close()

    sns.lineplot(x=df_ratio_unseen.index, y="train_size", data=df_ratio_unseen, label="Train set", marker="o")
    sns.lineplot(x=df_ratio_unseen.index, y="test_normal_size", data=df_ratio_unseen, label="Test normal", marker="o")
    sns.lineplot(x=df_ratio_unseen.index, y="test_abnormal_size", data=df_ratio_unseen, label="Test abnormal",
                 marker="o")
    plt.xticks(df_ratio_unseen.index)
    plt.title("Dataset size | " + name)
    plt.ylabel("Size")
    plt.xlabel("Data redistribution algorithm iterations")
    plt.legend()
    plt.savefig(output_dir + "ratio_unseen_dataset.png")
    plt.show()
    plt.close()

    sns.lineplot(x=df_ratio_unseen.index, y="train_vocab", data=df_ratio_unseen, label="Train", marker="o")
    sns.lineplot(x=df_ratio_unseen.index, y="test_normal_vocab", data=df_ratio_unseen, label="Test normal", marker="o")
    sns.lineplot(x=df_ratio_unseen.index, y="test_abnormal_vocab", data=df_ratio_unseen, label="Test abnormal",
                 marker="o")
    plt.xticks(df_ratio_unseen.index)
    plt.title("Vocab size | " + name)
    plt.ylabel("Size")
    plt.xlabel("Data redistribution algorithm iterations")
    plt.legend()
    plt.savefig(output_dir + "ratio_unseen_vocab.png")
    plt.show()
    plt.close()

    sns.lineplot(x=df_ratio_unseen.index, y="ratio_unseen_normal_sequences", data=df_ratio_unseen, label="Normal test sequences",
                 marker="o")
    sns.lineplot(x=df_ratio_unseen.index, y="ratio_unseen_abnormal_sequences", data=df_ratio_unseen, label="Abnormal test sequences",
                 marker="o")
    sns.lineplot(x=df_ratio_unseen.index, y="ratio_unseen_total_sequences", data=df_ratio_unseen, label="Total test sequences",
                 marker="o")
    plt.xticks(df_ratio_unseen.index)
    plt.title("Percentage of test sequences containing unseen logkeys | " + name)
    plt.ylabel("% unseen")
    plt.xlabel("Data redistribution algorithm iterations")
    plt.legend()
    plt.savefig(output_dir + "ratio_unseen_seqs.png")
    plt.show()
    plt.close()


def plot_train_size_comp(dataset_name, output_dir_corrected):
    df_seq_matrix_corrected = pd.read_csv(output_dir_corrected + "matrix/ratio_unseen.csv")

    df_seq_semantic = pd.read_csv("../output_parsing/" + dataset_name + "/semantic/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_seq_semantic["ratio"] = df_seq_matrix_corrected["ratio_unseen_normal_sequences"].apply(lambda x: round(x * 100, 2))

    df_seq_matrix = pd.read_csv("../output_parsing/" + dataset_name + "/matrix/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_seq_matrix["ratio"] = df_seq_matrix_corrected["ratio_unseen_normal_sequences"].apply(lambda x: round(x * 100, 2))

    df_seq_semantic_large_train = pd.read_csv("../output_big_train/" + dataset_name + "/semantic/results/Sequence Level.csv").iloc[
                                  1::2].reset_index(drop=True)
    df_seq_semantic_large_train["ratio"] = df_seq_matrix_corrected["ratio_unseen_normal_sequences"].apply(lambda x: round(x * 100, 2))

    df_seq_matrix_large_train = pd.read_csv("../output_big_train/" + dataset_name + "/matrix/results/Sequence Level.csv").iloc[1::2].reset_index(drop=True)
    df_seq_matrix_large_train["ratio"] = df_seq_matrix_corrected["ratio_unseen_normal_sequences"].apply(lambda x: round(x * 100, 2))

    sns.lineplot(x="ratio", y="MCC", data=df_seq_matrix, label="LogBERT Small Train", markers=True, marker="o",
                 markersize=7, color="C0")
    sns.lineplot(x="ratio", y="MCC", data=df_seq_matrix_large_train, label="LogBERT Large Train", markers=True, marker="s",
                 markersize=7, color="C0")
    sns.lineplot(x="ratio", y="MCC", data=df_seq_semantic, label="VoBERT Small Train", markers=True, marker="o",
                 markersize=7, color="C1")
    sns.lineplot(x="ratio", y="MCC", data=df_seq_semantic_large_train, label="VoBERT Large Train", markers=True,
                 marker="s", markersize=7, color="C1")
    # plt.xticks(df_seq_semantic["ratio"].apply(lambda x: round(x,0)))
    # print(df_seq_semantic["ratio"].apply(lambda x: round(x,0)))
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    plt.title("Performance on Sequence Level | " + dataset_name)
    plt.ylabel("MCC Score")
    plt.xlabel("% of normal sequences containing at least one unseen element")
    plt.legend()
    plt.savefig("../plots/performance_sequence_trainsize_comp_" + dataset_name + ".png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_ratio_unseen("../output/bgl/matrix/", "BGL")
    plot_ratio_unseen("../output/tbird/matrix/", "TBird")
    plot_ratio_unseen("../output/hdfs/matrix/", "HDFS")
    # plot_performance_vs_unseen("../output_parsing_corrected_unseen_stats/bgl/", "BGL", "../output/bgl/")
    # plot_performance_vs_unseen("../output_parsing_corrected_unseen_stats/tbird/", "TBird", "../output/tbird/")
    # plot_performance_vs_unseen("../output_parsing_corrected_unseen_stats/hdfs/", "HDFS", "../output/hdfs/")
    # plot_ratio_unseen_x_axis("./output/bgl/matrix/")
    # plot_train_valid_loss("./output/bgl/semantic/_0.0008/bert/", '_log2', "VF-MLM")
    # plot_train_valid_loss("./output/bgl/bert/", '_log2', "MLM")
    # plot_parsing_vs_no_parsing("BGL", "../output/bgl/")
    # plot_parsing_vs_no_parsing("TBird", "../output/tbird/")
    # plot_el_vs_seq("BGL")
    # plot_el_vs_seq("TBird")
    # plot_train_size_comp("BGL", "../output/bgl/")
    # plot_train_size_comp("TBird")
    # plot_train_size_comp("HDFS")
