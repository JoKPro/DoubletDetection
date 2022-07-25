import datatable as dt
import pandas as pd
import numpy as np
import gc
import scanpy

def filter_uniform_cols(data):
    nunique = data.nunique()
    to_drop = nunique[nunique == 1].index
    data.drop(to_drop, axis=1, inplace=True)
    return data

def class_checker(row):
    if row["homotypic"]:
        return "homo"
    elif row["heterotypic"]:
        return "hetero"
    else:
        return "singlet"


def prepare_data(expr_file, anno_file, use_reduction=False, reduction=0.8):
    print("loading data")
    expr = dt.fread(expr_file)
    anno = dt.fread(anno_file)
    print("data to pandas")
    anno = anno.to_pandas()
    expr = expr.to_pandas()

    print(anno)

    gc.collect()

    print("dropping columns")
    expr = expr.drop("C11", axis=1, errors="ignore")
    expr = expr.drop("C0", axis=1, errors="ignore")

    gc.collect()

    print("annotation")
    anno["cell_type"] = anno.apply(lambda row: class_checker(row), axis=1)
    anno_filtered = anno[["V1", "cell_type"]]
    anno_filtered.rename(columns={"V1": "C10"}, inplace=True)

    gc.collect()

    if use_reduction:
        print("using reduction")
        expr = expr.drop(
            expr.columns[np.random.choice(range(1, expr.shape[1]), size=int(reduction * expr.shape[1]), replace=False)],
            axis=1)
        expr = expr.drop(
            np.random.choice(range(1, expr.shape[0]), size=int(reduction * expr.shape[0]), replace=False), axis=0)
        print(f"new dimensions: {expr.shape}")

    print("merge")
    merged = pd.merge(expr, anno_filtered)
    print("deleting expr")
    del expr
    gc.collect()

    print("writing to file")
    merged.to_csv(f"../../data/processed/merged_filtered_{merged.shape[0]}_{merged.shape[1]}.csv")

    print(merged)
    print(f"class distribution: \n"
          f"{merged['cell_type'].value_counts()}")
    pass


def read_h5ad(expr_file, ann_file):
    full = scanpy.read_h5ad(expr_file)
    anno = pd.read_csv(ann_file)

    print(anno)
    anno["cell_type"] = anno.apply(lambda row: class_checker(row), axis=1)
    anno_filtered = anno[["V1", "cell_type"]]
    anno_filtered.rename(columns={"V1": "obs"}, inplace=True)
    print(anno_filtered)

    X = pd.DataFrame.sparse.from_spmatrix(full.X)
    X.columns = full.var_names
    X["obs"] = full.obs_names.str.replace(r'-', '.')

    print(X)

    merged = pd.merge(X, anno_filtered)
    merged.to_csv(f"../../data/processed/pbmc_hvg_{merged.shape[0]}_{merged.shape[1] - 2}.csv")
    print(merged.shape)
    return merged

prepare_data("../../data/raw/pbmc_expr_t.csv", "../../data/raw/Doubletannotation.csv")
# read_h5ad("../../data/raw/pbmc_500hvg.h5ad", "../../data/raw/Doubletannotation.csv")
# read_h5ad("../../data/raw/pbmc_1000hvg.h5ad", "../../data/raw/Doubletannotation.csv")
