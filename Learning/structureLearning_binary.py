import logging
import copy
import multiprocessing
import os
import time

from collections import deque
from enum import Enum
from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
from Learning.utils import convert_to_scope_domain, get_matached_domain
from Learning.statistics import get_structure_stats

logger = logging.getLogger(__name__)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import numpy as np

from Learning.transformStructure import Prune
from Learning.validity import is_valid
from Learning.splitting.RDC import rdc_test
from Structure.nodes import Product, Sum, Factorize, assign_ids

parallel = True

if parallel:
    cpus = max(1, os.cpu_count() - 2)
else:
    cpus = 1
pool = multiprocessing.Pool(processes=cpus)


def calculate_RDC(data, ds_context, scope, condition, sample_size):
    """
    Calculate the RDC adjacency matrix using the data
    """
    tic = time.time()
    scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
    meta_types = ds_context.get_meta_types_by_scope(scope_range)
    domains = ds_context.get_domains_by_scope(scope_range)

    # calculate the rdc scores, the parameter k to this function are taken from SPFlow original code
    if len(data) <= sample_size:
        rdc_adjacency_matrix = rdc_test(
            data, meta_types, domains, k=10
        )
    else:
        local_data_sample = data[np.random.randint(data.shape[0], size=sample_size)]
        rdc_adjacency_matrix = rdc_test(
            local_data_sample, meta_types, domains, k=10
        )
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    logging.debug(f"calculating pairwise RDC on sample {sample_size} takes {time.time() - tic} secs")
    return rdc_adjacency_matrix, scope_loc, condition_loc


class Operation(Enum):
    CREATE_LEAF = 1
    SPLIT_COLUMNS = 2
    SPLIT_ROWS = 3
    NAIVE_FACTORIZATION = 4  # This refers to consider the variables in the scope as independent
    REMOVE_UNINFORMATIVE_FEATURES = 5  # If all data of certain attribute are the same
    FACTORIZE = 6  # A factorized node
    REMOVE_CONDITION = 7  # Remove independent set from the condition
    SPLIT_ROWS_CONDITION = 8
    SPLIT_COLUMNS_CONDITION = 9  # NOT IMPLEMENTED Split rows when there is condition, using conditional independence
    FACTORIZE_CONDITION = 10  # NOT IMPLEMENTED Factorize columns when there is condition, using conditional independence


def get_next_operation(ds_context, min_instances_slice=100, min_features_slice=1, multivariate_leaf=True,
                       threshold=0.3, rdc_sample_size=50000, rdc_strong_connection_threshold=0.75):
    """
    :param ds_context: A context specifying the type of the variables in the dataset
    :param min_instances_slice: minimum number of rows to stop splitting by rows
    :param min_features_slice: minimum number of feature to stop splitting by columns usually 1
    :param multivariate_leaf: If true, we fit joint distribution with multivariates.
                              This only controls the SPN branch.
    :return: return a function, call to which will generate the next operation to perform
    """

    def next_operation(
            data,
            scope,
            condition,
            no_clusters=False,
            no_independencies=False,
            no_condition=False,
            is_strong_connected=False,
            rdc_threshold=threshold,
            rdc_strong_connection_threshold=rdc_strong_connection_threshold
    ):

        """
        :param data: local data set
        :param scope: scope of parent node
        :param condition: scope of conditional of parent node
        :param no_clusters: if True, we are assuming the highly correlated and unseparable data
        :param no_independencies: if True, cannot split by columns
        :param no_condition: if True, cannot remove conditions
        :param rdc_threshold: Under which, we will consider the two variables as independent
        :param rdc_strong_connection_threshold: Above which, we will consider the two variables as strongly related
        :return: Next Operation and parameters to call if have any
        """

        assert len(set(scope).intersection(set(condition))) == 0, "scope and condition mismatch"
        assert (len(scope) + len(condition)) == data.shape[1], "Redundant data columns"

        minimalFeatures = len(scope) == min_features_slice
        minimalInstances = data.shape[0] <= min_instances_slice

        if minimalFeatures and len(condition) == 0:
            return Operation.CREATE_LEAF, None

        if is_strong_connected and (len(condition) == 0):
            # the case of strongly connected components, directly model them
            return Operation.CREATE_LEAF, None

        if (minimalInstances and len(condition) == 0) or (no_clusters and len(condition) <= 1):
            if multivariate_leaf or is_strong_connected:
                return Operation.CREATE_LEAF, None
            else:
                return Operation.NAIVE_FACTORIZATION, None

        # Check if all data of an attribute has the same value (very possible for categorical data)
        uninformative_features_idx = np.var(data, 0) == 0
        ncols_zero_variance = np.sum(uninformative_features_idx)
        if ncols_zero_variance > 0:
            if ncols_zero_variance == data.shape[1]:
                if multivariate_leaf:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.NAIVE_FACTORIZATION, None
            else:
                feature_idx = np.asarray(sorted(scope + condition))
                uninformative_features = list(feature_idx[uninformative_features_idx])
                if set(uninformative_features) == set(scope):
                    if multivariate_leaf:
                        return Operation.CREATE_LEAF, None
                    else:
                        return Operation.NAIVE_FACTORIZATION, None
                if len(condition) == 0 or len(set(uninformative_features).intersection(set(condition))) != 0:
                    # This is very messy here but essentially realigning the scope and condition with the data column
                    return (
                        Operation.REMOVE_UNINFORMATIVE_FEATURES,
                        (get_matached_domain(uninformative_features_idx, scope, condition))
                    )

        if len(condition) != 0 and no_condition:
            """
                In this case, we have no condition to remove. Must split rows or create leaf.
            """
            if minimalInstances:
                if multivariate_leaf or is_strong_connected:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.NAIVE_FACTORIZATION, None
            elif not no_clusters:
                return Operation.SPLIT_ROWS_CONDITION, calculate_RDC(data, ds_context, scope, condition,
                                                                     rdc_sample_size)

        elif len(condition) != 0:
            """Try to eliminate some of condition, which are independent of scope
            """
            rdc_adjacency_matrix, scope_loc, condition_loc = calculate_RDC(data, ds_context, scope, condition,
                                                                           rdc_sample_size)
            independent_condition = []
            remove_cols = []
            for i in range(len(condition_loc)):
                cond = condition_loc[i]
                is_indep = True
                for s in scope_loc:
                    if rdc_adjacency_matrix[cond][s] > rdc_threshold:
                        is_indep = False
                        continue
                if is_indep:
                    remove_cols.append(cond)
                    independent_condition.append(condition[i])

            if len(independent_condition) != 0:
                return Operation.REMOVE_CONDITION, (independent_condition, remove_cols)

            else:
                # If there is nothing to eliminate from conditional set, we split rows
                if minimalInstances:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.SPLIT_ROWS_CONDITION, (rdc_adjacency_matrix, scope_loc, condition_loc)


        elif not no_clusters and not minimalInstances:
            """In this case:  len(condition) == 0 and not minimalFeatures and not no_clusters
               So we try to split rows or factorize
            """
            rdc_adjacency_matrix, scope_loc, _ = calculate_RDC(data, ds_context, scope, condition, rdc_sample_size)
            if not no_independencies:
                # test independence
                rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_threshold] = 0

                num_connected_comp = 0
                indep_res = np.zeros(data.shape[1])
                for i, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix))):
                    indep_res[list(c)] = i + 1
                    num_connected_comp += 1
                if num_connected_comp > 1:
                    # there exists independent sets, split by columns
                    return Operation.SPLIT_COLUMNS, indep_res

            rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_strong_connection_threshold] = 0
            strong_connected_comp = []  # strongly connected components
            for c in connected_components(from_numpy_matrix(rdc_adjacency_matrix)):
                if len(c) > 1:
                    component = list(c)
                    component.sort()
                    for i in range(len(c)):
                        component[i] = scope[component[i]]
                    strong_connected_comp.append(component)

            if len(strong_connected_comp) != 0:
                if strong_connected_comp[0] == scope:
                    # the whole scope is actually strongly connected
                    return Operation.CREATE_LEAF, None
                # there exists sets of strongly connect component, must factorize them out
                return Operation.FACTORIZE, strong_connected_comp

        elif minimalInstances:
            if multivariate_leaf or is_strong_connected:
                return Operation.CREATE_LEAF, None
            else:
                return Operation.NAIVE_FACTORIZATION, None

        # if none of the above conditions follows, we split by row and try again.
        if len(condition) == 0:
            return Operation.SPLIT_ROWS, None
        else:
            return Operation.SPLIT_ROWS_CONDITION, calculate_RDC(data, ds_context, scope, condition, rdc_sample_size)

    return next_operation


def default_slicer(data, cols, num_cond_cols=None):
    if num_cond_cols is None:
        if len(cols) == 1:
            return data[:, cols[0]].reshape((-1, 1))

        return data[:, cols]
    else:
        return np.concatenate((data[:, cols], data[:, -num_cond_cols:]), axis=1)


def learn_structure_binary(
        dataset,
        ds_context,
        split_rows,
        split_rows_condition,
        split_cols,
        create_leaf,
        create_leaf_multi,
        threshold,
        rdc_sample_size,
        next_operation=None,
        min_row_ratio=0.01,
        rdc_strong_connection_threshold=0.75,
        multivariate_leaf=True,
        initial_scope=None,
        data_slicer=default_slicer,
        debug=True
):
    assert dataset is not None
    assert ds_context is not None
    assert split_rows is not None
    assert split_cols is not None
    assert create_leaf is not None
    assert create_leaf_multi is not None

    if next_operation == None:
        if min_row_ratio < 1:
            min_row = int(min_row_ratio * dataset.shape[0])
        else:
            min_row = min_row_ratio
        next_operation = get_next_operation(ds_context, min_row,
                                            threshold=threshold, rdc_sample_size=rdc_sample_size,
                                            rdc_strong_connection_threshold=rdc_strong_connection_threshold,
                                            multivariate_leaf=multivariate_leaf)

    root = Product()
    root.children.append(None)

    if initial_scope is None:
        initial_scope = list(range(dataset.shape[1]))
        initial_cond = []
        num_conditional_cols = None
    elif len(initial_scope) < dataset.shape[1]:
        num_conditional_cols = dataset.shape[1] - len(initial_scope)
        initial_cond = [item for item in list(range(dataset.shape[1])) if item not in initial_scope]
    else:
        num_conditional_cols = None
        initial_cond = []
        assert len(initial_scope) > dataset.shape[1], "check initial scope: %s" % initial_scope

    tasks = deque()
    tasks.append((dataset, root, 0, initial_scope, initial_cond, None, False, False, False, False))

    while tasks:

        local_data, parent, children_pos, scope, condition, rect_range, no_clusters,\
        no_independencies, no_condition, is_strong_connected = tasks.popleft()

        if debug:
            logging.debug(f"Current task with data {local_data.shape} scope {scope} and condition {condition}")
        operation, op_params = next_operation(
            local_data,
            scope,
            condition,
            no_clusters=no_clusters,
            no_independencies=no_independencies,
            no_condition=no_condition,
            is_strong_connected=is_strong_connected
        )

        if debug:
            logging.debug("OP: {} on slice {} (remaining tasks {})".format(operation, local_data.shape, len(tasks)))

        if operation == Operation.REMOVE_UNINFORMATIVE_FEATURES:
            # Very messy because of the realignment from scope domain, condition domain and data column domain.
            (scope_rm, scope_rm2, scope_keep, condition_rm, condition_keep) = op_params
            new_condition = [condition[i] for i in condition_keep]
            keep_all = [item for item in range(local_data.shape[1]) if item not in condition_rm + scope_rm]

            if len(new_condition) != len(condition) and debug:
                logging.debug(
                    f"find uninformation condition, keeping only condition {new_condition}")
            if len(new_condition) != 0:
                # only condition variables have been removed
                keep_all = [item for item in range(local_data.shape[1]) if item not in condition_rm]
                assert (len(scope) + len(new_condition)) == len(
                    keep_all), f"Redundant data columns, {scope}, {new_condition}, {keep_all}"
                tasks.append(
                    (
                        data_slicer(local_data, keep_all, num_conditional_cols),
                        parent,
                        children_pos,
                        scope,
                        new_condition,
                        rect_range,
                        no_clusters,
                        no_independencies,
                        True,
                        is_strong_connected
                    )
                )
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
            else:
                # we need to create product node if scope variables have been removed
                node = Product()
                node.scope = copy.deepcopy(scope)
                node.condition = copy.deepcopy(new_condition)
                node.range = copy.deepcopy(rect_range)
                parent.children[children_pos] = node

                rest_scope = copy.deepcopy(scope)
                for i in range(len(scope_rm)):
                    col = scope_rm[i]
                    new_scope = scope[scope_rm2[i]]
                    rest_scope.remove(new_scope)
                    node.children.append(None)
                    assert col not in keep_all
                    if debug:
                        logging.debug(
                            f"find uninformative scope {new_scope}")
                    tasks.append(
                        (
                            data_slicer(local_data, [col], num_conditional_cols),
                            node,
                            len(node.children) - 1,
                            [new_scope],
                            [],
                            rect_range,
                            True,
                            True,
                            True,
                            False
                        )
                    )

                next_final = False

                if len(rest_scope) == 0:
                    continue
                elif len(rest_scope) == 1:
                    next_final = True

                node.children.append(None)
                c_pos = len(node.children) - 1

                if debug:
                    logging.debug(
                        f"The rest scope {rest_scope} and condition {new_condition} keep"
                    )
                    assert (len(rest_scope) + len(new_condition)) == len(keep_all), "Redundant data columns"
                tasks.append(
                    (
                        data_slicer(local_data, keep_all, num_conditional_cols),
                        node,
                        c_pos,
                        rest_scope,
                        new_condition,
                        rect_range,
                        next_final,
                        next_final,
                        False,
                        is_strong_connected
                    )
                )
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.REMOVE_CONDITION:
            (independent_condition, remove_cols) = op_params
            new_condition = [item for item in condition if item not in independent_condition]
            keep_cols = [item for item in range(local_data.shape[1]) if item not in remove_cols]
            if debug:
                logging.debug(
                    f"Removed uniformative condition {independent_condition}")
                assert (len(scope) + len(new_condition)) == len(keep_cols), "Redundant data columns"
            tasks.append(
                (
                    data_slicer(local_data, keep_cols, num_conditional_cols),
                    parent,
                    children_pos,
                    scope,
                    new_condition,
                    rect_range,
                    no_clusters,
                    no_independencies,
                    True,
                    is_strong_connected
                )
            )
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.SPLIT_ROWS_CONDITION:

            split_start_t = perf_counter()
            data_slices = split_rows_condition(local_data, ds_context, scope, condition, op_params)
            split_end_t = perf_counter()

            if debug:
                logging.debug(
                    "\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
                )


            if len(data_slices) == 1:
                tasks.append((local_data, parent, children_pos, scope, condition,
                              rect_range, True, False, False, is_strong_connected))
                continue

            node = Sum()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            parent.children[children_pos] = node
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            for data_slice, range_slice, proportion, _ in data_slices:
                assert (len(scope) + len(condition)) == data_slice.shape[1], "Redundant data columns"
                node.children.append(None)
                node.weights.append(proportion)
                new_rect_range = dict()
                for c in rect_range:
                    if c not in range_slice:
                        new_rect_range[c] = rect_range[c]
                    else:
                        new_rect_range[c] = range_slice[c]
                tasks.append((data_slice, node, len(node.children) - 1, scope, condition,
                              new_rect_range, False, False, False, is_strong_connected))
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.SPLIT_ROWS:

            split_start_t = perf_counter()
            data_slices = split_rows(local_data, ds_context, scope)
            split_end_t = perf_counter()

            if debug:
                logging.debug(
                    "\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
                )

            if len(data_slices) == 1:
                tasks.append((local_data, parent, children_pos, scope, condition,
                              rect_range, False, True, False, is_strong_connected))
                continue

            node = Sum()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            parent.children[children_pos] = node

            for data_slice, scope_slice, proportion in data_slices:
                assert isinstance(scope_slice, list), "slice must be a list"
                assert (len(scope) + len(condition)) == data_slice.shape[1], "Redundant data columns"
                node.children.append(None)
                node.weights.append(proportion)
                tasks.append((data_slice, node, len(node.children) - 1, scope, condition,
                              rect_range, False, False, False,
                              is_strong_connected))
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.SPLIT_COLUMNS:
            split_start_t = perf_counter()
            data_slices = split_cols(local_data, ds_context, scope, clusters=op_params)
            split_end_t = perf_counter()

            if debug:
                logging.debug(
                    "\t\tfound {} col clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
                )

            if len(data_slices) == 1:
                tasks.append((local_data, parent, children_pos, scope, condition,
                              rect_range, False, True, False, is_strong_connected))
                assert np.shape(data_slices[0][0]) == np.shape(local_data)
                assert data_slices[0][1] == scope
                continue

            node = Product()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            parent.children[children_pos] = node

            for data_slice, scope_slice, _ in data_slices:
                assert isinstance(scope_slice, list), "slice must be a list"
                assert (len(scope_slice) + len(condition)) == data_slice.shape[1], "Redundant data columns"
                node.children.append(None)
                if debug:
                    logging.debug(
                        f'Create an independent component with scope {scope_slice} and condition {condition}'
                    )
                tasks.append((data_slice, node, len(node.children) - 1, scope_slice, condition,
                              rect_range, False, True, False,
                              is_strong_connected))
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.FACTORIZE:
            # condition should be [] when we do factorize
            node = Factorize()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            parent.children[children_pos] = node
            index_list = sorted(scope + condition)

            # if there are multiple components we left it for the next round
            if debug:
                for comp in op_params:
                    logging.debug(
                        f'Factorize node found the strong connected component{comp}'
                    )
                logging.debug(
                    f'We only factor out {op_params[0]}'
                )

            strong_connected = op_params[0]
            other_connected = [item for item in scope if item not in strong_connected]

            assert len(other_connected) != 0, "factorize results in only one strongly connected"
            node.children.append(None)
            data_copy = copy.deepcopy(local_data)
            if debug:
                logging.debug(
                    f'Factorize node factor out weak connected component{other_connected}'
                )
            keep_cols = [index_list.index(i) for i in sorted(other_connected + condition)]
            tasks.append(
                (
                    data_slicer(data_copy, keep_cols, num_conditional_cols),
                    node,
                    0,
                    other_connected,
                    condition,
                    rect_range,
                    False,
                    False,
                    False,
                    False
                )
            )
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            new_condition = sorted(condition + other_connected)
            node.children.append(None)
            new_scope = strong_connected
            keep_cols = [index_list.index(i) for i in sorted(new_scope + new_condition)]
            if debug:
                logging.debug(
                    f'Factorize node found a strongly connect component{new_scope}, '
                    f'condition on {new_condition}'
                )
                assert (len(new_scope) + len(new_condition)) == len(keep_cols), "Redundant data columns"
            if rect_range is None:
                new_rect_range = dict()
            else:
                new_rect_range = copy.deepcopy(rect_range)
            for i, c in enumerate(new_condition):
                condition_idx = []
                for j in new_condition:
                    condition_idx.append(index_list.index(j))
                data_attr = local_data[:, condition_idx[i]]
                new_rect_range[c] = [(np.nanmin(data_attr), np.nanmax(data_attr))]
            tasks.append(
                (
                    data_slicer(local_data, keep_cols, num_conditional_cols),
                    node,
                    1,
                    new_scope,
                    new_condition,
                    new_rect_range,
                    False,
                    True,
                    False,
                    True
                )
            )
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.NAIVE_FACTORIZATION:
            # This is assuming the remaining attributes as independent. FSPN will probably never get here.
            node = Product()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            parent.children[children_pos] = node

            scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
            local_tasks = []
            local_children_params = []
            split_start_t = perf_counter()
            for i, col in enumerate(scope_loc):
                node.children.append(None)
                local_tasks.append(len(node.children) - 1)
                child_data_slice = data_slicer(local_data, [col], num_conditional_cols)
                local_children_params.append((child_data_slice, ds_context, [scope[i]], []))

            result_nodes = pool.starmap(create_leaf, local_children_params)

            for child_pos, child in zip(local_tasks, result_nodes):
                node.children[child_pos] = child

            split_end_t = perf_counter()

            logging.debug(
                "\t\tnaive factorization {} columns (in {:.5f} secs)".format(len(scope), split_end_t - split_start_t)
            )
            continue

        elif operation == Operation.CREATE_LEAF:
            leaf_start_t = perf_counter()
            if len(scope) == 1:
                node = create_leaf(local_data, ds_context, scope, condition)
            else:
                node = create_leaf_multi(local_data, ds_context, scope, condition)
            node.range = rect_range
            parent.children[children_pos] = node
            leaf_end_t = perf_counter()

            logging.debug(
                "\t\t created leaf {} for scope={} and condition={} (in {:.5f} secs)".format(
                    node.__class__.__name__, scope, condition, leaf_end_t - leaf_start_t
                )
            )
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue
        else:
            raise Exception("Invalid operation: " + operation)

    node = root.children[0]
    assign_ids(node)
    print(get_structure_stats(node))
    valid, err = is_valid(node)
    assert valid, "invalid spn: " + err
    node = Prune(node)
    valid, err = is_valid(node)
    assert valid, "invalid spn: " + err

    return node
