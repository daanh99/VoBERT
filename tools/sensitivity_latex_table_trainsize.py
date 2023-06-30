import pandas as pd


def get_performance(output_dir, stability):
    df_seq = pd.read_csv(output_dir + "results/Sequence Level.csv").iloc[1::2]
    df_seq["ratio"] = df_seq["ratio"].apply(lambda x: round(x * 100, 2))
    if stability == "Mean":
        mean_MCC = df_seq["MCC"].mean()
        mean_F1 = df_seq["F1"].mean()
        mean_AUPRC = df_seq["AUROC"].mean()
        return mean_MCC, mean_F1, mean_AUPRC
    else:
        row_seq = df_seq[df_seq["ratio"] == df_seq["ratio"].min()] if stability == "No" else df_seq[
            df_seq["ratio"] == df_seq["ratio"].max()]
        MCC = row_seq["MCC"].item()
        F1 = row_seq["F1"].item()
        AUPRC = row_seq["AUROC"].item()
        return MCC, F1, AUPRC


def generate_train_size_table():
    datasets = ["BGL"]
    methods = {"VF-MLM": "semantic", "MLM": "matrix"}
    stabilities = ["No", "Yes", "Mean"]
    table = "\\begin{tabular}{lll|lll|lll}\n"
    table += "\\hline\n"
    table += "\\multicolumn{3}{l}{} & \\multicolumn{3}{c}{Small Train} & \\multicolumn{3}{c}{Large Train}  \\\\\n"
    table += "Dataset & Method & Instability & F1 & MCC & AUPRC & F1 & MCC & AUPRC \\\\\n"
    table += "\\hline\n"

    for dataset in datasets:
        row_count = 0
        for method, folder in methods.items():
            for i, stability in enumerate(stabilities):
                row_count += 1
                train_size = "Min" if stability == "No" else ("Max" if stability == "Yes" else "Mean")
                output_dir_small = f"../output_parsing/{dataset.lower()}/{folder}/"
                output_dir_large = f"../output/{dataset.lower()}/{folder}/"
                MCC_small, F1_small, AUPRC_small = get_performance(output_dir_small, stability)
                MCC_large, F1_large, AUPRC_large = get_performance(output_dir_large, stability)
                row_small = f" & {f'{F1_small:.2f}'} & {f'{MCC_small:.2f}'} & {f'{AUPRC_small:.2f}'}"
                row_large = f" & {f'{F1_large:.2f}'} & {f'{MCC_large:.2f}'} & {f'{AUPRC_large:.2f}'} \\\\"

                if row_count == 1:
                    table += "\\multirow{6}{*}{" + dataset + "} & \\multirow{3}{*}{" + method + "} & " + train_size + row_small + row_large + "\n"
                elif row_count == 4:
                    table += " & \\multirow{3}{*}{" + method + "} & " + train_size + row_small + row_large + "\n"
                else:
                    table += " & & " + train_size + row_small + row_large + "\n"
        table += "\\hline\n"
    table += "\\end{tabular}"
    return table


if __name__ == "__main__":
    print(generate_train_size_table())
