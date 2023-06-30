import pandas as pd

def get_performance(output_dir, stability):
    df_seq = pd.read_csv(output_dir + "semantic/results/Sequence Level.csv").iloc[1::2]
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


def generate_sensitivity_table():
    datasets = ["BGL", "TBird"]
    method = "VF-MLM"
    stabilities = ["No", "Yes", "Mean"]
    table = "\\begin{tabular}{lll|lll|lll}\n"
    table += "\\hline\n"
    table += "\\multicolumn{3}{l}{} & \\multicolumn{3}{c}{Parsing (Sequence-level)} & \\multicolumn{3}{c}{No Parsing (Sequence-level)}  \\\\\n"
    table += "Dataset & Method & Instability & F1 & MCC & AUPRC & F1 & MCC & AUPRC \\\\\n"
    table += "\\hline\n"
    for dataset in datasets:
        for i, stability in enumerate(stabilities):
            parsing_value = "Min" if stability == "No" else ("Max" if stability == "Yes" else "Mean")
            output_dir = f"../output_parsing/{dataset.lower()}/"
            no_parse_output_dir = f"../output_no_parsing/{dataset.lower()}/"
            MCC, F1, AUPRC = get_performance(output_dir, stability)
            MCC_no_parse, F1_no_parse, AUPRC_no_parse = get_performance(no_parse_output_dir, stability)
            row = f" & {f'{F1:.2f}'} & {f'{MCC:.2f}'} & {f'{AUPRC:.2f}'}"
            row_no_parse = f" & {f'{F1_no_parse:.2f}'} & {f'{MCC_no_parse:.2f}'} & {f'{AUPRC_no_parse:.2f}'} \\\\"

            if i == 0:
                table += "\\multirow{3}{*}{" + dataset + "} & \\multirow{3}{*}{" + method + "} & " + parsing_value + row + row_no_parse + "\n"
            else:
                table += " & & " + parsing_value + row + row_no_parse + "\n"
        table += "\\hline\n"
    table += "\\end{tabular}"
    return table



if __name__ == "__main__":
    print(generate_sensitivity_table())
