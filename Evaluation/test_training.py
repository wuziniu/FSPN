import logging
import argparse
import time
import sys
import pandas as pd
sys.path.append('../../fspn')

from Learning.learningWrapper import learn_FSPN, learn_FSPN_binary
from Structure.nodes import Context
from Structure.leaves.parametric.Parametric import Categorical, Gaussian
from Evaluation.toy_dataset import *
from Structure.model import FSPN
from Structure.StatisticalTypes import MetaType


def test_on_toy(dataset_name, nrows=100000, seed=0):
    if dataset_name == "strong_corr_cat":
        data = toy_data_highly_correlated_cat(nrows=nrows, seed=seed)
        ds_context = Context(parametric_types=[Categorical, Categorical, Categorical, Categorical,
                                               Categorical, Categorical, Categorical, Categorical]).add_domains(data)
    elif dataset_name == "weak_corr_cat":
        data = toy_data_slightly_correlated_cat(nrows=nrows, seed=seed)
        ds_context = Context(parametric_types=[Categorical, Categorical, Categorical, Categorical,
                                               Categorical, Categorical, Categorical, Categorical]).add_domains(data)
    elif dataset_name == "independent_cat":
        data = toy_data_independent_cat(nrows=nrows, seed=seed)
        print(type(data), data.shape)
        ds_context = Context(parametric_types=[Categorical, Categorical, Categorical, Categorical,
                                               Categorical, Categorical, Categorical, Categorical]).add_domains(data)

    elif dataset_name == "strong_corr_cont":
        data = toy_data_highly_correlated_cont(nrows=nrows, seed=seed)
        ds_context = Context(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian,
                                               Gaussian, Gaussian, Gaussian, Gaussian]).add_domains(data)
    elif dataset_name == "weak_corr_cont":
        data = toy_data_slightly_correlated_cont(nrows=nrows, seed=seed)
        ds_context = Context(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian,
                                               Gaussian, Gaussian, Gaussian, Gaussian]).add_domains(data)
    elif dataset_name == "independent_cont":
        data = toy_data_independent_cont(nrows=nrows, seed=seed)
        ds_context = Context(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian,
                                               Gaussian, Gaussian, Gaussian, Gaussian]).add_domains(data)

    else:
        raise NotImplemented

    logging.info("Loading data completed, start the training process")

    tic = time.time()
    model = FSPN()
    fspn = learn_FSPN(data, ds_context, rdc_sample_size=100000,
                         rdc_strong_connection_threshold=0.7)
    model.model = fspn
    model.store_factorize_as_dict()
    logging.info(f"FSPN training complete, takes {time.time()-tic} seconds")
    return model

def get_ds_context_discrete(data):
    #All columns are categorical
    context = []
    for i in range(data.shape[1]):
        context.append(MetaType.BINARY)
    return Context(meta_types=context).add_domains(data)

def test_on_binary(dataset_name):
    directory_path = '/Users/ziniuwu/Desktop/research/data_predictions_scripts/datasets/discrete/'
    file_name = directory_path+dataset_name+'.ts.csv'
    df = pd.read_csv(file_name, sep=',', header=None)
    data = df.values
    print(data.shape)
    logging.info("Loading data completed, start the training process")

    ds_context = get_ds_context_discrete(data)
    tic = time.time()
    model = FSPN()
    fspn = learn_FSPN_binary(data, ds_context, rdc_strong_connection_threshold=0.7)
    model.model = fspn
    logging.info(f"FSPN training complete, takes {time.time() - tic} seconds")
    return fspn




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='toy', type=str, help='Which type of experiment to run')
    parser.add_argument('--dataset', default='independent_cat', type=str, help='Which dataset to be used')
    parser.add_argument('--nrows', default=100000, type=int, help='How many lines to create')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--log_level', type=int, default=logging.DEBUG)
    parser.add_argument('--ingore_warning', action='store_true')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    if args.ingore_warning:
        import warnings
        warnings.filterwarnings("ignore")

    #fspn = FSPN()
    if args.experiment == 'toy':
        fspn = test_on_toy(args.dataset, args.nrows, args.seed)
    elif args.experiment == 'binary':
        fspn = test_on_binary(args.dataset)