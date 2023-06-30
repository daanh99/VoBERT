import pandas as pd


def mean_performance(output_dir, el_output_dir):
    df_seq = pd.read_csv(output_dir + "results/Sequence Level.csv").iloc[1::2]
    df_seq["ratio"] = df_seq["ratio"].apply(lambda x: round(x * 100, 2))

    mean_MCC = df_seq["MCC"].mean()
    mean_F1 = df_seq["F1"].mean()
    mean_AUPRC = df_seq["AUROC"].mean()

    df_el = pd.read_csv(el_output_dir + "results/Element Level.csv").iloc[1::2]
    df_aso = pd.read_csv(el_output_dir + "results/Element Level ASO.csv").iloc[1::2]

    if df_el.empty or df_aso.empty:
        mean_MCC_el, mean_F1_el, mean_AUPRC_el, mean_MCC_aso, mean_F1_aso, mean_AUPRC_aso = '-', '-', '-', '-', '-', '-'
    else:
        mean_MCC_el = df_el["MCC"].mean()
        mean_F1_el = df_el["F1"].mean()
        mean_AUPRC_el = df_el["AUROC"].mean()

        mean_MCC_aso = df_aso["MCC"].mean()
        mean_F1_aso = df_aso["F1"].mean()
        mean_AUPRC_aso = df_aso["AUROC"].mean()

    return mean_MCC, mean_F1, mean_AUPRC, mean_MCC_el, mean_F1_el, mean_AUPRC_el, mean_MCC_aso, mean_F1_aso, mean_AUPRC_aso


def performance(output_dir, el_output_dir, unstable):
    df_seq = pd.read_csv(output_dir + "results/Sequence Level.csv").iloc[1::2]
    df_el = pd.read_csv(el_output_dir + "results/Element Level.csv").iloc[1::2]
    df_aso = pd.read_csv(el_output_dir + "results/Element Level ASO.csv").iloc[1::2]

    df_seq["ratio"] = df_seq["ratio"].apply(lambda x: round(x * 100, 2))

    if df_el.empty or df_aso.empty:
        MCC_el, F1_el, AUPRC_el, MCC_aso, F1_aso, AUPRC_aso = '-', '-', '-', '-', '-', '-'
    else:
        if unstable == "No":
            row_el = df_el[df_el["ratio"] == df_el["ratio"].min()]
            row_aso = df_aso[df_aso["ratio"] == df_aso["ratio"].min()]
        else:  # unstable == "Yes"
            row_el = df_el[df_el["ratio"] == df_el["ratio"].max()]
            row_aso = df_aso[df_aso["ratio"] == df_aso["ratio"].max()]

        MCC_el = row_el["MCC"].item()
        F1_el = row_el["F1"].item()
        AUPRC_el = row_el["AUROC"].item()

        MCC_aso = row_aso["MCC"].item()
        F1_aso = row_aso["F1"].item()
        AUPRC_aso = row_aso["AUROC"].item()

    if unstable == "No":
        row_seq = df_seq[df_seq["ratio"] == df_seq["ratio"].min()]
    else:  # unstable == "Yes"
        row_seq = df_seq[df_seq["ratio"] == df_seq["ratio"].max()]

    MCC = row_seq["MCC"].item()
    F1 = row_seq["F1"].item()
    AUPRC = row_seq["AUROC"].item()

    return MCC, F1, AUPRC, MCC_el, F1_el, AUPRC_el, MCC_aso, F1_aso, AUPRC_aso

def generate_latex_table_new():
    datasets = ["HDFS", "BGL", "TBird"]
    methods = {"MLM": "matrix", "VF-MLM": "semantic", "Heuristic": "heuristic"}
    stabilities = ["No", "Yes", "Mean"]

    table = "\\begin{tabular}{lll|lll|lll|lll}\n"
    table += "\\hline\n"
    table += "\\multicolumn{3}{l}{} & \\multicolumn{3}{c}{Sequence-level} & \\multicolumn{3}{c}{Element-level} & \\multicolumn{3}{c}{Element-level ASO}  \\\\\n"
    table += "Dataset & Method & Unstable & F1 & MCC & AUPRC & F1 & MCC & AUPRC & F1 & MCC & AUPRC \\\\\n"
    table += "\\hline\n"

    for dataset in datasets:
        table += "\\multirow{9}{*}{" + dataset + "}\n"
        for method, folder in methods.items():
            first_method = True
            for stability in stabilities:
                if first_method:
                    table += " & \\multirow{3}{*}{" + method + "}"
                else:
                    table += " &"
                table += " & " + stability
                output_dir = f"../output_parsing/{dataset.lower()}/{folder}/"
                el_output_dir = f"../output_el_level/{dataset.lower()}/{folder}/"
                if stability == "Mean":
                    MCC, F1, AUPRC, MCC_el, F1_el, AUPRC_el, MCC_aso, F1_aso, AUPRC_aso = mean_performance(output_dir,
                                                                                                           el_output_dir)
                else:
                    MCC, F1, AUPRC, MCC_el, F1_el, AUPRC_el, MCC_aso, F1_aso, AUPRC_aso = performance(output_dir,
                                                                                                      el_output_dir,
                                                                                                      stability)

                # Check types for MCC_el, F1_el, AUPRC_el, MCC_aso, F1_aso, AUPRC_aso
                if isinstance(MCC_el, str):
                    row_el = f" & {MCC_el} & {F1_el} & {AUPRC_el}"
                else:
                    row_el = f" & {f'{F1_el:.2f}'} & {f'{MCC_el:.2f}'} & {f'{AUPRC_el:.2f}'}"

                if isinstance(MCC_aso, str):
                    row_aso = f" & {MCC_aso} & {F1_aso} & {AUPRC_aso}"
                else:
                    row_aso = f" & {f'{F1_aso:.2f}'} & {f'{MCC_aso:.2f}'} & {f'{AUPRC_aso:.2f}'}"

                row = f" & {f'{F1:.2f}'} & {f'{MCC:.2f}'} & {f'{AUPRC:.2f}'}"
                table += row + row_el + row_aso + " \\\\\n"
                first_method = False
            table += "\\cline{2-12}\n"
        table += "\\hline\n"
    table += "\\end{tabular}"
    return table



if __name__ == "__main__":
    print(generate_latex_table_new())
