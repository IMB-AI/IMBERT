"""Turn per-sample pathway abundance tables into the IMBERT input format.

Input
-----
One tab-separated file per sample inside `raw_input_dir/DL_input_<version>/`,
named `<SampleId>_*`, with the two columns produced by HUMAnN:

    Unnamed: 0                              <SampleId>
    PWY-6990: (+)-camphor biosynthesis      0.0123
    ...

Output
------
`preprocessing_dir/<version>/DL_input_<version>.tsv`: one row per pathway, one
column per sample, holding the *rank* of that pathway inside the sample
(1 = most abundant, 0 = absent).  Pathway names are normalised to
`PATHWAY NAME:PWY-ID` so that they match the released IMBERT vocabulary.

Usage
-----
    python preprocessing.py --config preprocessing_config.ini
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from configobj import ConfigObj

# Greek letters are written as HTML-ish entities in some HUMAnN releases.
_ABNORMAL_PWY_LIST = [f"&{name}" for name in
                      ['alpha', 'beta', 'delta', 'gamma', 'iota', 'kappa', 'lambda', 'omega']]


def normalise_pathway_names(pathway_series):
    """`PWY-1234: [Some] alpha-name` -> `ALPHA-NAME:PWY-1234`."""
    revised_pwy_name = []
    for pwy in pathway_series:
        pwy_id = pwy.split(": ")[0]
        pwy_name = pwy.split(": ")[-1]
        for abnormal_pwy in _ABNORMAL_PWY_LIST:
            if abnormal_pwy in pwy_name:
                pwy_name = pwy_name.replace(abnormal_pwy, abnormal_pwy.replace("&", ""))
        if pwy_name.startswith("["):
            pwy_name = pwy_name.split("] ") [-1]
        revised_pwy_name.append(pwy_name.upper() + ":" + pwy_id)

    return revised_pwy_name


def rank_one_sample(path, tmp_dir=None):
    """Read one abundance file and return a single-column ranked DataFrame."""
    sample = os.path.basename(path).split("_")[0]
    df = pd.read_csv(path, sep="\t").rename(columns={sample: "Abundance"})
    df["Unnamed: 0"] = normalise_pathway_names(df["Unnamed: 0"])

    sorted_pathways = sorted(df["Unnamed: 0"].tolist())

    # Rank the non-zero pathways; the most abundant pathway gets rank 1.
    non_zero_df = df.loc[df["Abundance"] > 0, :].copy()
    ranks = non_zero_df.rank(method="max", ascending=False)[["Abundance"]]
    non_zero_df = pd.concat([non_zero_df, ranks.rename(columns={"Abundance": sample})], axis=1)

    if tmp_dir:
        tmp_output = pd.concat([df.loc[df["Abundance"] == 0, :], non_zero_df]).fillna(0)
        tmp_output = tmp_output.set_index("Unnamed: 0").loc[sorted_pathways, :].reset_index()
        tmp_output.to_csv(os.path.join(tmp_dir, f"{sample}_pwy_contents_with_rank.tsv"),
                          sep="\t", index=False)

    # `rank(method="max")` gives every member of a tie the same rank; spread
    # those ties over consecutive integers so that no two pathways of a sample
    # share a position in the sequence.
    tie_counts = (non_zero_df.value_counts(sample) > 1).to_frame()
    tied_ranks = tie_counts.loc[tie_counts["count"], :].index.tolist()

    revised_tied_df = None
    if tied_ranks:
        tied_df = non_zero_df.loc[non_zero_df[sample].isin(tied_ranks), :].sort_values(by=sample)
        revised_rows = []
        for rank in tied_ranks:
            current = tied_df.loc[tied_df[sample] == rank, :].sort_values(by="Unnamed: 0").copy()
            current[sample] = np.arange(rank - len(current) + 1, rank + 1, 1)
            revised_rows.append(current.loc[:, ["Unnamed: 0", "Abundance", sample]])
        revised_tied_df = pd.concat(revised_rows)

        if tmp_dir:
            tied_df.to_csv(os.path.join(tmp_dir, f"{sample}_equal_ranks.tsv"),
                           sep="\t", index=False)

    zero_df = df.loc[df["Abundance"] == 0, :].copy()
    zero_df[sample] = 0
    untied_df = non_zero_df.loc[~non_zero_df[sample].isin(tied_ranks), :]

    parts = [zero_df, untied_df]
    if revised_tied_df is not None:
        parts.append(revised_tied_df)
    final_df = pd.concat(parts).set_index("Unnamed: 0").loc[sorted_pathways, :]

    if tmp_dir:
        final_df.to_csv(os.path.join(tmp_dir, f"{sample}_processed_DLinput.tsv"), sep="\t")

    return final_df.loc[:, [sample]]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=str,
                        help="path to preprocessing_config.ini")
    parser.add_argument("--save-tmp", action="store_true",
                        help="also write the per-sample intermediate tables")
    args = parser.parse_args()

    config = ConfigObj(args.config, file_error=True)
    input_dir = os.path.join(config["raw_input_dir"], "DL_input_" + config["version"])
    output_dir = os.path.join(config["preprocessing_dir"], config["version"])
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = None
    if args.save_tmp:
        tmp_dir = os.path.join(output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

    data_path_list = sorted(glob.glob(os.path.join(input_dir, "*")))
    if not data_path_list:
        raise FileNotFoundError(f"no abundance file found in {input_dir}")

    output_df_list = []
    for path_idx, path in enumerate(data_path_list):
        if path_idx % 100 == 0:
            print(f"{path_idx}/{len(data_path_list)} samples processed")
        output_df_list.append(rank_one_sample(path, tmp_dir))

    output_path = os.path.join(output_dir, "DL_input_" + config["version"] + ".tsv")
    pd.concat(output_df_list, axis=1).to_csv(output_path, sep="\t")
    print(f"{len(output_df_list)} samples written to {output_path}")


if __name__ == "__main__":
    main()
