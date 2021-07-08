import os
import pandas as pd
import numpy as np

np.random.seed(1)


def toy_data_highly_correlated_cont(schema=None, path=None, nrows=500000, return_df=False, seed=0):
    """
    Create some highly correlated toy table for evaluation and debug purposes
    """
    np.random.seed(seed)
    if schema is not None:
        table_obj = schema.table_dictionary['strong_corr']

    #generate attributes
    attr1 = np.random.normal(0, 20, size=nrows)
    attr2 = np.random.normal(0, 20, size=nrows)
    attr3 = attr1 + attr2
    attr4 = attr1 * 2.5 + 5
    attr5 = attr2 * 2.5 + 5
    attr6 = attr1 ** 2 / 100 + attr5
    attr7 = attr1 + attr4 + np.random.normal(0, 2, size=nrows)
    attr8 = attr3 + attr5 + attr6 + attr7

    if path is not None and not os.path.exists(path):
        os.makedirs(path)

    if return_df:
        # return pandas dataframe
        dataset = pd.DataFrame({'id': np.arange(nrows), 'cont_attr1': attr1, 'cont_attr2': attr2, 'cont_attr3': attr3,
                                'cont_attr4': attr4,
                                'cont_attr5': attr5, 'cont_attr6': attr6, 'cont_attr7': attr7, 'cont_attr8': attr8})

        if schema is not None:
            from Data_prepare.prepare_single_tables import prepare_all_tables
            dataset.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]
            dataset = dataset.apply(pd.to_numeric, errors="ignore")

            prepare_all_tables(schema, path, loaded=dataset)
            print(f"Files successfully created")

        return dataset


    else:
        # return numpy matrix
        dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8]

    return dataset


def toy_data_slightly_correlated_cont(schema=None, path=None, nrows=500000, return_df=False, seed=0):
    """
    Create some slightly correlated toy table for evaluation and debug purposes
    """
    np.random.seed(seed)
    if schema is not None:
        table_obj = schema.table_dictionary['weak_corr']

    attr1 = np.random.randint(10, size=nrows)
    attr2 = np.random.randint(3, size=nrows)
    attr3 = attr1 + attr2 + np.random.choice([0, 1], size=nrows, p=[0.8, 0.2])
    attr4 = np.random.normal(10, 20, size=nrows)
    attr5 = attr4 * 2.5 + np.random.normal(0, 3, size=nrows)
    attr6 = np.random.randint(10, size=nrows) ** 2 + np.random.randint(3, size=nrows) * attr4 / 100
    attr7 = attr4 / 2 + attr1 / 2 + np.random.normal(0, 3, size=nrows)
    attr8 = attr3 * 2 + attr4 + np.random.normal(0, 3, size=nrows)

    if path is not None and not os.path.exists(path):
        os.makedirs(path)

    if return_df:
        # return pandas dataframe
        dataset = pd.DataFrame({'id': np.arange(nrows), 'cont_attr1': attr1, 'cont_attr2': attr2, 'cont_attr3': attr3,
                                'cont_attr4': attr4,
                                'cont_attr5': attr5, 'cont_attr6': attr6, 'cont_attr7': attr7, 'cont_attr8': attr8})

        if schema is not None:
            from Data_prepare.prepare_single_tables import prepare_all_tables
            dataset.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]
            dataset = dataset.apply(pd.to_numeric, errors="ignore")

            prepare_all_tables(schema, path, loaded=dataset)
            print(f"Files successfully created")

        return dataset


    else:
        # return numpy matrix
        dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8]

    return dataset


def toy_data_independent_cont(schema=None, path=None, nrows=500000, return_df=False, seed=0):
    """
    Create some independent toy table for evaluation and debug purposes
    """
    np.random.seed(seed)
    if schema is not None:
        table_obj = schema.table_dictionary['independent']

    attr1 = np.random.randint(10, size=nrows)
    attr2 = np.random.randint(3, size=nrows)
    attr3 = np.random.randint(10, size=nrows) + np.random.randint(3, size=nrows)
    attr4 = np.random.normal(10, 20, size=nrows)
    attr5 = np.random.normal(10, 20, size=nrows) * 2.5
    attr6 = np.random.randint(3, size=nrows) * np.random.normal(10, 20, size=nrows)
    attr7 = np.random.normal(10, 20, size=nrows) / 2 + np.random.randint(10, size=nrows)
    attr8 = np.random.randint(10, size=nrows) * 2 + np.random.normal(10, 20, size=nrows)

    if path is not None and not os.path.exists(path):
        os.makedirs(path)

    if return_df:
        # return pandas dataframe
        dataset = pd.DataFrame({'id': np.arange(nrows), 'cont_attr1': attr1, 'cont_attr2': attr2, 'cont_attr3': attr3,
                                'cont_attr4': attr4,
                                'cont_attr5': attr5, 'cont_attr6': attr6, 'cont_attr7': attr7, 'cont_attr8': attr8})

        if schema is not None:
            from Data_prepare.prepare_single_tables import prepare_all_tables
            dataset.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]
            dataset = dataset.apply(pd.to_numeric, errors="ignore")

            prepare_all_tables(schema, path, loaded=dataset)
            print(f"Files successfully created")

        return dataset


    else:
        # return numpy matrix
        dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8]

    return dataset


def toy_data_highly_correlated_cat(schema=None, path=None, nrows=500000, return_df=False, seed=0):
    """
    Create some highly correlated toy table for evaluation and debug purposes
    """
    np.random.seed(seed)
    if schema is not None:
        table_obj = schema.table_dictionary['strong_corr']

    attr1 = np.random.randint(10, size=nrows)
    attr2 = attr1 + np.random.choice([0, 1], size=nrows, p=[0.9, 0.1])
    attr3 = attr1 + attr2
    attr4 = attr3 + np.random.choice([0, 1, 2], size=nrows, p=[0.8, 0.1, 0.1])
    attr5 = attr3 + attr4
    attr6 = attr2 * 2 + np.random.randint(10, size=nrows)
    attr7 = attr1 + 10
    attr8 = attr1 + attr4 + attr5 + attr7

    if path is not None and not os.path.exists(path):
        os.makedirs(path)

    if return_df:
        # return pandas dataframe
        dataset = pd.DataFrame({'id': np.arange(nrows), 'cont_attr1': attr1, 'cont_attr2': attr2, 'cont_attr3': attr3,
                                'cont_attr4': attr4,
                                'cont_attr5': attr5, 'cont_attr6': attr6, 'cont_attr7': attr7, 'cont_attr8': attr8})

        if schema is not None:
            from Data_prepare.prepare_single_tables import prepare_all_tables
            dataset.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]
            dataset = dataset.apply(pd.to_numeric, errors="ignore")

            prepare_all_tables(schema, path, loaded=dataset)
            print(f"Files successfully created")

        return dataset

    else:
        # return numpy matrix
        dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8]

    return dataset

def toy_data_medium_correlated_cat(schema=None, path=None, nrows=500000, return_df=False, seed=0):
    """
    Create some highly correlated toy table for evaluation and debug purposes
    """
    np.random.seed(seed)
    if schema is not None:
        table_obj = schema.table_dictionary['strong_corr']

    attr1 = np.random.randint(10, size=nrows)
    attr2 = attr1 + np.random.choice([0, 1], size=nrows, p=[0.9, 0.1])
    attr3 = attr1 + attr2
    attr4 = attr3 + np.random.choice([0, 1, 2], size=nrows, p=[0.8, 0.1, 0.1])
    attr5 = np.random.randint(10, size=nrows)
    attr6 = attr5 + np.random.choice([0, 1, 2], size=nrows, p=[0.5, 0.2, 0.3])
    attr7 = attr4 + attr5 + np.random.randint(5, size=nrows)
    attr8 = attr4 + np.random.randint(10, size=nrows)
    attr9 = np.random.randint(20, size=nrows)
    attr10 = np.random.randint(20, size=nrows) + np.random.choice([0, 1, 2], size=nrows, p=[0.4, 0.3, 0.3])

    if path is not None and not os.path.exists(path):
        os.makedirs(path)

    if return_df:
        # return pandas dataframe
        dataset = pd.DataFrame({'id': np.arange(nrows), 'cont_attr1': attr1, 'cont_attr2': attr2, 'cont_attr3': attr3,
                                'cont_attr4': attr4, 'cont_attr5': attr5, 'cont_attr6': attr6, 'cont_attr7': attr7,
                                'cont_attr8': attr8, 'cont_attr9': attr9, 'cont_attr10': attr10})

        if schema is not None:
            from Data_prepare.prepare_single_tables import prepare_all_tables
            dataset.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]
            dataset = dataset.apply(pd.to_numeric, errors="ignore")

            prepare_all_tables(schema, path, loaded=dataset)
            print(f"Files successfully created")

        return dataset

    else:
        # return numpy matrix
        dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10]

    return dataset

def toy_data_slightly_correlated_cat(schema=None, path=None, nrows=500000, return_df=False, seed=0):
    """
    Create some slightly correlated toy table for evaluation and debug purposes
    """
    np.random.seed(seed)
    if schema is not None:
        table_obj = schema.table_dictionary['weak_corr']

    attr1 = np.random.randint(10, size=nrows)
    attr2 = np.random.randint(10, size=nrows)
    attr3 = attr1 + attr2 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr4 = attr1 + np.random.choice([0, 1, 2], size=nrows, p=[0.7, 0.2, 0.1])
    attr5 = np.random.randint(40, size=nrows)
    attr6 = attr5 + attr4 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr7 = attr2 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr8 = np.random.randint(100, size=nrows)

    if path is not None and not os.path.exists(path):
        os.makedirs(path)

    if return_df:
        # return pandas dataframe
        dataset = pd.DataFrame({'id': np.arange(nrows), 'cont_attr1': attr1, 'cont_attr2': attr2, 'cont_attr3': attr3,
                                'cont_attr4': attr4,
                                'cont_attr5': attr5, 'cont_attr6': attr6, 'cont_attr7': attr7, 'cont_attr8': attr8})

        if schema is not None:
            from Data_prepare.prepare_single_tables import prepare_all_tables
            dataset.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]
            dataset = dataset.apply(pd.to_numeric, errors="ignore")

            prepare_all_tables(schema, path, loaded=dataset)
            print(f"Files successfully created")

        return dataset

    else:
        # return numpy matrix
        dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8]

    return dataset

def toy_data_independent_cat(schema=None, path=None, nrows=500000, return_df=False, seed=0):
    """
    Create some independent toy table for evaluation and debug purposes
    """
    np.random.seed(seed)
    if schema is not None:
        table_obj = schema.table_dictionary['independent']

    attr1 = np.random.randint(10, size=nrows)
    attr2 = np.random.randint(11, size=nrows)
    attr3 = np.random.randint(21, size=nrows) + np.random.randint(3, size=nrows)
    attr4 = np.random.randint(24, size=nrows)
    attr5 = np.random.randint(45, size=nrows)
    attr6 = np.random.randint(13, size=nrows)
    attr7 = np.random.randint(58, size=nrows)
    attr8 = np.random.randint(137, size=nrows)

    if path is not None and not os.path.exists(path):
        os.makedirs(path)

    if return_df:
        # return pandas dataframe
        dataset = pd.DataFrame({'id': np.arange(nrows), 'cont_attr1': attr1, 'cont_attr2': attr2, 'cont_attr3': attr3,
                                'cont_attr4': attr4,
                                'cont_attr5': attr5, 'cont_attr6': attr6, 'cont_attr7': attr7, 'cont_attr8': attr8})

    else:
        # return numpy matrix
        dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8]

    return dataset

def toy_data_totally_corr_iclr(nrows=1000000, seed=0):
    """
    Create some independent toy table for evaluation and debug purposes
    """
    np.random.seed(seed)

    attr1 = np.random.randint(4, size=nrows)
    attr2 = attr1 + 1
    attr3 = attr1 + attr2
    attr4 = attr1 * 2 + 1
    attr5 = attr2 * 3 - 1
    attr6 = attr5 + attr4
    attr7 = attr6 - attr3
    attr8 = attr2 ** 2
    attr9 = attr8 + attr7 + attr5 + attr3
    attr10 = attr9 - attr6 - attr4 - attr2 - attr1

    dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10]
    return dataset


def toy_data_highly_corr_iclr(nrows=1000000, seed=0):
    """
    Create some independent toy table for evaluation and debug purposes
    """
    np.random.seed(seed)

    attr1 = np.random.randint(3, size=nrows)
    attr2 = attr1 + np.random.choice([0, 1], size=nrows, p=[0.8, 0.2])
    attr3 = attr1 + np.random.choice([0, 1, 2], size=nrows, p=[0.7, 0.2, 0.1])
    attr4 = attr1 * 2 + np.random.choice([0, 1], size=nrows, p=[0.8, 0.2])
    attr5 = attr2 + attr3
    attr6 = np.random.randint(3, size=nrows)
    attr7 = attr6 + np.random.choice([0, 1], size=nrows, p=[0.9, 0.1])
    attr8 = attr6 + attr1
    attr9 = attr6 * 2 + np.random.choice([0, 1], size=nrows, p=[0.8, 0.2]) + attr1
    attr10 = attr7 + attr9

    dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10]
    return dataset


def toy_data_medium_corr_iclr(nrows=1000000, seed=0):
    """
    Create some independent toy table for evaluation and debug purposes
    """
    np.random.seed(seed)

    attr1 = np.random.randint(3, size=nrows)
    attr2 = attr1 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr3 = attr2 + np.random.choice([0, 1, 2], size=nrows, p=[0.6, 0.2, 0.2])
    attr4 = attr1 * 2 + np.random.choice([0, 1], size=nrows, p=[0.8, 0.2])
    attr5 = attr2 + attr3 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr6 = np.random.randint(3, size=nrows)
    attr7 = attr6 + np.random.choice([0, 1], size=nrows, p=[0.9, 0.1])
    attr8 = attr6 + attr1
    attr9 = attr6 * 2 + np.random.choice([0, 1], size=nrows, p=[0.8, 0.2]) + attr1
    attr10 = attr7 + attr9

    dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10]
    return dataset

def toy_data_slightly_corr_iclr(nrows=1000000, seed=0):
    """
    Create some independent toy table for evaluation and debug purposes
    """
    
    attr1 = np.random.randint(3, size=nrows)
    attr2 = np.random.randint(3, size=nrows)
    attr3 = attr1 + attr2 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr4 = attr1 * 2 + np.random.choice([0, 1, 2], size=nrows, p=[0.7, 0.2, 0.1])
    attr5 = attr2 + attr3 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr6 = np.random.randint(3, size=nrows)
    attr7 = attr6 + attr4 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr8 = attr2 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr9 = attr6 + np.random.choice([0, 1], size=nrows, p=[0.7, 0.3])
    attr10 = attr8 + attr9

    dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10]
    return dataset


def toy_data_independent_iclr(nrows=1000000, seed=0):
    """
    Create some independent toy table for evaluation and debug purposes
    """
    np.random.seed(seed)

    attr1 = np.random.randint(4, size=nrows)
    attr2 = np.random.randint(4, size=nrows) + 1
    attr3 = np.random.randint(4, size=nrows) + 2
    attr4 = np.random.randint(4, size=nrows) * 2
    attr5 = np.random.randint(4, size=nrows) * 3
    attr6 = (np.random.randint(4, size=nrows) + 1) * 3 - 1
    attr7 = np.random.randint(4, size=nrows) ** 2
    attr8 = np.random.randint(4, size=nrows) * 2 + 1
    attr9 = np.random.randint(4, size=nrows)
    attr10 = np.random.randint(4, size=nrows)

    dataset = np.c_[attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10]
    return dataset
