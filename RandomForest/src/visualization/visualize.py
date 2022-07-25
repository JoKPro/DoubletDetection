import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import itertools
import pprint as pp


def prepare_data_for_plots(prediction, labels):
    classes = np.unique(labels)
    predictions = prediction["prediction"] ##irrelevant
    report = classification_report(y_true=labels, y_pred=predictions, labels=classes, output_dict=True)
    metrics = {key: report[key] for key in ["singlet", "hetero", "homo"]}

    return classes, predictions, report, metrics


def make_conf_matrix(y_probs, labels, project_name, naive=True):

    if naive:
        predictions = y_probs.idxmax(axis=1)

    cnf_matrix = confusion_matrix(y_true=labels, y_pred=predictions, labels=np.unique(labels))
    class_rep = classification_report(y_true=labels, y_pred=predictions, labels=np.unique(labels), output_dict=True)
    print(cnf_matrix)
    print(classification_report(y_true=labels, y_pred=predictions, labels=np.unique(labels)))
    pp.pprint(class_rep)
    print(class_rep)
    print(pd.DataFrame(class_rep))
    # pd.DataFrame(class_rep).to_csv("../../reports/500_metric_table.csv", float_format="%.2f")
    plt.imshow(cnf_matrix, cmap=plt.cm.Blues)
    threshold = cnf_matrix.max() / 2
    for i in range(cnf_matrix.shape[0]):
        for j in range(cnf_matrix.shape[1]):
            plt.text(j, i, cnf_matrix[i,j], color="w" if cnf_matrix[i, j] > threshold else "black")
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.colorbar()
    plt.tight_layout()

    result_dir = f"../../reports/figures/{project_name}_cnf_mat"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    plt.savefig(f"{result_dir}/{project_name}_conf_mat.png")
    pass

# yan_labels = pd.read_csv("../../data/processed/yannick_labels.csv")
# y_probs_yan = pd.read_csv("../../data/processed/yannick_predictions.csv")#.drop("prediction", axis=1)
# labels_yan = pd.read_csv("../../data/processed/yannick_labels.csv")["cell_type"]

# y_probs_100 = pd.read_csv("../../models/pbmc_hvg_100/prediction.csv", index_col=0).drop("prediction", axis=1)
# labels_100 = pd.read_csv("../../models/pbmc_hvg_100/test_set.csv", index_col=0)["cell_type"]

y_probs_500 = pd.read_csv("../../models/pbmc_hvg_500/prediction.csv", index_col=0).drop("prediction", axis=1)
labels_500 = pd.read_csv("../../models/pbmc_hvg_500/test_set.csv", index_col=0)["cell_type"]

# y_probs_1000 = pd.read_csv("../../models/pbmc_hvg_1000/prediction.csv", index_col=0).drop("prediction", axis=1)
# labels_1000 = pd.read_csv("../../models/pbmc_hvg_1000/test_set.csv", index_col=0)["cell_type"]

make_conf_matrix(y_probs=y_probs_500, labels=labels_500, project_name="testst")

def get_reps(projects):
    pass


def plot_roc(fpr, tpr, thresholds, ax):
    g = sns.lineplot(x=fpr, y=tpr, ax=ax)
    g.axline(xy1=(0,0), slope=1, color="r", dashes=(5,2))


def plot_roc_projects(proj_fpr_tpr, cls, ax):
    data_sets = []
    legend_handle = []
    for project_name in proj_fpr_tpr.keys():
        dat = pd.DataFrame( {"fpr": proj_fpr_tpr[project_name]["fpr"][cls],
                             "tpr": proj_fpr_tpr[project_name]["tpr"][cls]})
        dat["Run"] = project_name
        data_sets.append(dat)
        legend_handle.append(f"Run: {project_name} - AUC_micro: {proj_fpr_tpr[project_name]['auc'][cls]:.2f}")

    data = pd.concat(data_sets, ignore_index=True)
    g = sns.lineplot(x="fpr", y="tpr", data=data, ax=ax, hue="Run")
    g.axline(xy1=(0,0), slope=1, color="grey", dashes=(5,2))
    ax.legend(legend_handle)


def make_roc_curves_OvR_projects(projects, results_dir, project_name, classifier):
    """Make ROC figure from multiple runs. Input must be a dictionary containing the project name
    as keys and dictionaries containing the test set (label probs) and actual labels"""

    proj_fpr_tpr = dict()
    classes = ["hetero", "homo", "singlet"]
    fig = plt.figure(figsize=(12, 6))

    for project_name in projects.keys():
        labels = projects[project_name]["labels"]
        print(projects[project_name]["prediction"])
        y_proba = projects[project_name]["prediction"].drop("prediction", axis=1, errors="ignore")

        print(y_proba)
        print(f"y_proba : \n"
              f"{y_proba}")
        print(f"labels: \n"
              f"{labels}")

        fpr_dict = dict()
        tpr_dict = dict()
        auc_dict = dict()
        for i in range(len(classes)):

            c = classes[i]

            df_aux = pd.DataFrame(y_proba.copy(deep=True))
            df_aux['class'] = [1 if y == c else 0 for y in np.array(labels)]

            print("--------------------------_")
            print(y_proba)
            print(y_proba)
            print(y_proba.shape)
            print(df_aux.shape)
            print(df_aux)
            print(c)
            df_aux['prob'] = y_proba.loc[:, c]
            df_aux = df_aux.reset_index(drop=True)
            fpr, tpr, thresholds = roc_curve(df_aux['class'], df_aux['prob'])
            tpr_dict.update({c: tpr})
            fpr_dict.update({c: fpr})
            auc_dict.update({c:  roc_auc_score(df_aux["class"], df_aux["prob"], average="micro")})

        proj_fpr_tpr.update({project_name: {"fpr": fpr_dict, "tpr": tpr_dict, "auc": auc_dict}})

    for i in range(len(classes)):
        c = classes[i]


        if c == "hetero":
            cls_full = "Heterotypic Doublets"
        elif c == "homo":
            cls_full = "Homotypic Doublets"
        else:
            cls_full = "Single Cells"

        ax_bottom = fig.add_subplot(1, 3, i+1)
        ax_bottom.set_aspect("equal")
        plot_roc_projects(proj_fpr_tpr, ax=ax_bottom, cls=c)
        ax_bottom.set_title(cls_full)
        ax_bottom.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_bottom.set_ylabel("True Positive Rate (Sensitivity)")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle(f"{classifier} classification results:\n"
                 f"One v. Rest", size="x-large")

    result_dir = f"../../reports/figures/{results_dir}"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    fig.savefig(f"{result_dir}/{project_name}_roc_curves_OvR.png")
    fig.show()

def make_roc_curves_OvO_projects(projects, results_dir, project_name, classifier):
    proj_fpr_tpr = dict()
    classes = ["hetero", "homo", "singlet"]
    fig = plt.figure(figsize=(10, 17))
    cls_combs = list(itertools.permutations(classes, 2))
    print(cls_combs)

    for project_name in projects.keys():
        labels = projects[project_name]["labels"]
        print(projects[project_name]["prediction"])
        y_proba = projects[project_name]["prediction"].drop("prediction", axis=1, errors="ignore")

        print(f"y_proba : \n"
              f"{y_proba}")


        fpr_dict = dict()
        tpr_dict = dict()
        auc_dict = dict()
        for i in range(len(cls_combs)):
            comb = cls_combs[i]
            c1 = comb[0]
            c2 = comb[1]
            c1c2 = f"{c1} vs. {c2}"

            df_aux = pd.DataFrame(y_proba.copy(deep=True))
            df_aux['class'] = labels
            print("------------------")
            print(df_aux)
            df_aux['prob'] = y_proba.loc[:, c1]

            df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
            df_aux["class"] = [1 if y == c1 else 0 for y in df_aux["class"]]
            df_aux = df_aux.reset_index(drop = True)

            fpr, tpr, thresholds = roc_curve(df_aux['class'], df_aux['prob'])
            tpr_dict.update({c1c2: tpr})
            fpr_dict.update({c1c2: fpr})
            auc_dict.update({c1c2:  roc_auc_score(df_aux["class"], df_aux["prob"], average="micro")})

        proj_fpr_tpr.update({project_name: {"fpr": fpr_dict, "tpr": tpr_dict, "auc": auc_dict}})
        proj_fpr_tpr.update

    for i in range(len(cls_combs)):
        c = cls_combs[i]
        c1 = c[0]
        c2 = c[1]

        title = f"{c1} vs. {c2}"

        ax_bottom = fig.add_subplot(3, 2, i + 1)
        ax_bottom.set_aspect("equal")
        plot_roc_projects(proj_fpr_tpr, ax=ax_bottom, cls=title)
        ax_bottom.set_title(title)
        ax_bottom.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_bottom.set_ylabel("True Positive Rate (Sensitivity)")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle(f"{classifier} classification results:\n"
                 f"One v. One" , size="xx-large", y=0.95)

    result_dir = f"../../reports/figures/{results_dir}"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    fig.savefig(f"{result_dir}/{project_name}_roc_curves_OvO.png")
    fig.show()



    proj_fpr_tpr.update({project_name: {"fpr": fpr_dict, "tpr": tpr_dict}})





project_names = ["pbmc_hvg_100",
                 "pbmc_hvg_500",
                 "pbmc_hvg_1000"]

projects = {
    project_name: {
        "labels": pd.read_csv(f"../../models/{project_name}/test_set.csv", index_col=0)["cell_type"],
        "prediction": pd.read_csv(f"../../models/{project_name}/prediction.csv", index_col=0)
    } for project_name in project_names
}

project_yannick = {
    "FNN": {
        "labels": pd.read_csv("../../data/processed/yannick_labels.csv"),
        "prediction": pd.read_csv("../../data/processed/yannick_predictions.csv")
    }
}

make_roc_curves_OvR_projects(projects, "comparisons_auc_debug", "ovr", "Random Forest")
# make_roc_curves_OvO_projects(project_yannick, "yannick", "ovo", "FNN")



def make_roc_curves(project_yannick, actual, project_name):
    labels, predictions, reports, metrics = prepare_data_for_plots(prediction, actual)

    y_proba = prediction.drop("prediction", axis=1)
    fig = plt.figure()
    bins = [i / 20 for i in range(20)] + [1]
    classes = labels
    roc_auc_ovr = {}

    print(classes)
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]



        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame(predictions.copy(deep=True))
        df_aux['class'] = [1 if y == c else 0 for y in np.array(actual)]
        df_aux['prob'] = y_proba.iloc[:, i]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 3, i + 1)
        sns.histplot(x="prob", data=df_aux, hue='class', color='b', ax=ax, bins=bins)
        ax.set_title(c)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 3, i + 4)
        roc_auc_ovr = roc_auc_score(df_aux['class'], df_aux['prob'])
        fpr, tpr, thresholds = roc_curve(df_aux['class'], df_aux['prob'])
        plot_roc(fpr, tpr, thresholds, ax=ax_bottom)
        ax_bottom.set_title(cls_full)
        ax_bottom.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_bottom.set_ylabel("True Positive Rate (Sensitivity)")

        # Calculates the ROC AUC OvR
        roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle(y=0.5, size="xx-large", t="Random Forest classification results: \n"
              "More features have slight positive impact on model quality (AUC)")
    fig.show()

