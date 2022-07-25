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