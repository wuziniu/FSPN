import numpy as np
import time
import logging
from Structure.nodes import Context, Sum, Product, Factorize, Leaf, get_nodes_by_type,\
    get_topological_order, get_parents
from Structure.StatisticalTypes import MetaType
from Structure.leaves.fspn_leaves.Merge_leaves import Merge_leaves
from Learning.validity import is_valid
from Learning.learningWrapper import learn_FSPN
from Inference.inference import prod_likelihood, sum_likelihood, prod_log_likelihood, sum_log_likelihood


logger = logging.getLogger(__name__)

def get_ds_context_discrete(data):
    #All columns are categorical
    context = []
    for i in range(data.shape[1]):
        context.append(MetaType.BINARY)
    return Context(meta_types=context).add_domains(data)


def get_ds_context_categorical(data):
    #All columns are categorical
    context = []
    for i in range(data.shape[1]):
        context.append(MetaType.DISCRETE)
    return Context(meta_types=context).add_domains(data)


def build_ds_context(column_names, meta_types, null_values, table_meta_data, no_compression_scopes, train_data,
                     group_by_threshold=1200):
    """
    Builds context according to training data.
    :param column_names:
    :param meta_types:
    :param null_values:
    :param table_meta_data:
    :param train_data:
    :return:
    """
    ds_context = Context(meta_types=meta_types)
    ds_context.null_values = null_values
    assert ds_context.null_values is not None, "Null-Values have to be specified"
    domain = []
    no_unique_values = []
    # If metadata is given use this to build domains for categorical values
    unified_column_dictionary = None
    if table_meta_data is not None:
        unified_column_dictionary = {k: v for table, table_md in table_meta_data.items() if
                                     table != 'inverted_columns_dict' and table != 'inverted_fd_dict'
                                     for k, v in table_md['categorical_columns_dict'].items()}

    # domain values
    group_by_attributes = []
    for col in range(train_data.shape[1]):

        feature_meta_type = meta_types[col]
        min_val = np.nanmin(train_data[:, col])
        max_val = np.nanmax(train_data[:, col])

        unique_vals = len(np.unique(train_data[:, col]))
        no_unique_values.append(unique_vals)
        if column_names is not None:
            if unique_vals <= group_by_threshold and 'mul_' not in column_names[col] and '_nn' not in column_names[col]:
                group_by_attributes.append(col)

        domain_values = [min_val, max_val]

        if feature_meta_type == MetaType.REAL:
            min_val = np.nanmin(train_data[:, col])
            max_val = np.nanmax(train_data[:, col])
            domain.append([min_val, max_val])
        elif feature_meta_type == MetaType.DISCRETE:
            # if no metadata is given, infer domains from data
            if column_names is not None \
                    and unified_column_dictionary.get(column_names[col]) is not None:
                no_diff_values = len(unified_column_dictionary[column_names[col]].keys())
                domain.append(np.arange(0, no_diff_values + 1, 1))
            else:
                domain.append(np.arange(domain_values[0], domain_values[1] + 1, 1))
        else:
            raise Exception("Unkown MetaType " + str(meta_types[col]))

    ds_context.domains = np.asanyarray(domain)
    ds_context.no_unique_values = np.asanyarray(no_unique_values)
    ds_context.group_by_attributes = group_by_attributes

    if no_compression_scopes is None:
        no_compression_scopes = []
    ds_context.no_compression_scopes = no_compression_scopes

    return ds_context

def merge_leaves(node):
    """
    Convert a product node of only leaf children into a multivariate leaf for faster inference
    :param node: a product node
    :return: a leaf node
    """
    assert isinstance(node, Product), "incorrect parent node"
    scope = []
    for leaf in node.children:
        assert isinstance(leaf, Leaf), f"invalid children node type {type(leaf)}"
        scope.extend(leaf.scope)
    assert set(scope) == set(node.scope), "unmatched scope"
    new_node = Merge_leaves(node.children, node.scope, ranges=node.range)
    return new_node


class FSPN:
    def __init__(self):
        self.model = None
        self.ds_context = None

        # training stats
        self.learn_time = None
        self.rdc_threshold = None
        self.min_instances_slice = None
        self.pre_calculated = None


    def learn_from_data(self, train_data, ds_context, rdc_threshold=0.3, min_instances_slice=1,
                        max_sampling_threshold_cols=50000):

        # build domains (including the dependence analysis)
        learn_start_t = time.perf_counter()
        if min_instances_slice <= 1:
            min_instances_slice = round(len(train_data)*min_instances_slice)

        self.model = learn_FSPN(train_data, ds_context, threshold=rdc_threshold,
                                   rdc_sample_size=max_sampling_threshold_cols)

        assert is_valid(self.model, check_ids=True)
        learn_end_t = time.perf_counter()
        self.learn_time = learn_end_t - learn_start_t
        logging.debug(f"Built SPN in {learn_end_t - learn_start_t} sec")

        # statistics
        self.rdc_threshold = rdc_threshold
        self.min_instances_slice = min_instances_slice

    def store_factorize_as_dict(self):
        """
        1. Store the factorize node in a dictionary like data structure with key = id
        2. Store all ranges of leave node w.r.t. a right branch of factorize node as array.
        """
        self.fact_node = dict()
        for fact_node in get_nodes_by_type(self.model, Factorize):
            self.fact_node[fact_node.id] = fact_node

        self.leaves = dict()
        self.leaves_condition = dict()
        self.leaves_range = dict()
        self.weak_connected_leaves = []
        parents = get_parents(self.model)
        for fact_id in self.fact_node:
            node = self.fact_node[fact_id]
            assert len(node.children) == 2, "invalid fspn"
            right_branch = node.children[1]
            assert right_branch.range is not None, "right branch of a fact node has no range"
            for r_prod in get_nodes_by_type(right_branch, Product):
                #merge the product node in right branch to a single leaf node
                (parent_node, pos) = parents[r_prod][0]
                new_leaf = merge_leaves(r_prod)
                parent_node.children[pos] = new_leaf
                del r_prod
            leave_condition = []
            for key in right_branch.range:
                leave_condition.append(key)
            leave = []
            leave_left_bound = []
            leave_right_bound = []
            for r_leaf in get_nodes_by_type(right_branch, Leaf):
                leave.append(r_leaf)
                assert r_leaf.range is not None, "right branch leaf of a fact node has no range"
                left_bound = []
                right_bound = []
                for attr in r_leaf.range:
                    lrange = r_leaf.range[attr]
                    if type(lrange[0]) == tuple:
                        left_bound.append(lrange[0][0])
                        right_bound.append(lrange[0][1])
                    elif len(lrange) == 1:
                        left_bound.append(lrange[0])
                        right_bound.append(lrange[0])
                    else:
                        left_bound.append(lrange[0])
                        right_bound.append(lrange[1])
                leave_left_bound.append(left_bound)
                leave_right_bound.append(right_bound)
            self.leaves_condition[fact_id] = leave_condition
            self.leaves[fact_id] = leave
            self.leaves_range[fact_id] = (np.asarray(leave_left_bound), np.asarray(leave_right_bound))

        for leaf in get_nodes_by_type(self.model, Leaf):
            is_weak = True
            for fact_id in self.leaves:
                if leaf in self.leaves[fact_id]:
                    is_weak = False
            if is_weak:
                self.weak_connected_leaves.append(leaf)


    def get_overlap(self, a, b):
        """
        Calculate the overlap of l1 and l2
        :param a: of shape (np.array(n,k), np.array(n,k))
        :param b: of shape (np.array(m,k), np.array(m,k))
        :return: l: of shape (np.array(m,n,k), np.array(m,n,k))
        _ can have multiple dimensions
        """
        al, ar = a
        bl, br = b
        (n, k) = al.shape
        left_res = np.zeros((bl.shape[0], n, k))
        right_res = np.zeros((br.shape[0], n, k))
        for i in range(n):
            left_res[:, i, :] = np.maximum(bl, al[i, :])
            right_res[:, i, :] = np.minimum(br, ar[i, :])
        return left_res.reshape((-1, k)), right_res.reshape((-1, k))

    def _probability_left_most(self, query, node, attr):
        """
            calculate the probability on spn without factorized node
        """
        nodes = get_topological_order(node)

        all_results = {}

        for n in nodes:
            if isinstance(n, Leaf):
                result = n.query(query, attr)
            else:
                tmp_children_list = []
                for i in range(len(n.children)):
                    ci = n.children[i]
                    tmp_children_list.append(all_results[ci])
                if isinstance(n, Sum):
                    result = sum_likelihood(n, tmp_children_list)
                elif isinstance(n, Product):
                    result = prod_likelihood(n, tmp_children_list)
                else:
                    assert not isinstance(n, Factorize), "Factorize node should be eliminated"

            all_results[n] = result

        return all_results[node]

    def _spn_probability(self, query, node, query_attr, calculated):
        """
            Calculate the probability of a branch with all factorized node evaluated
            Node must contain a factorize node as children in this case.
        """
        assert query[0].shape[-1] == len(query_attr)
        if isinstance(node, Leaf):
            return node.query(query, query_attr)
        elif isinstance(node, Factorize):
            return calculated[node.id]

        child_res = []
        for child in node.children:
            child_res.append(self._spn_probability(query, child, query_attr, calculated))
        if isinstance(node, Sum):
            return sum_likelihood(node, child_res)
        else:
            return prod_likelihood(node, child_res)


    def _leave_prob(self, query, fact_id, attr):
        """
            calculate a batch of range probability on leaves
            query of shape (n,k)
            output shape (n,m) where m is the number of leaves of a fact_node
        """
        leaves = self.leaves[fact_id]
        probs = np.zeros((len(query[0]), len(leaves)))
        for i, leaf in enumerate(leaves):
            probs[:, i] = leaf.query(query, attr)
        return probs


    def probability(self, query, node=None, query_attr=None, calculated=dict()):
        """
        Calculate the probability
        :param query: two numpy arrays of value refers to the lower and upper range of each attribute
        :param node: start the evaluation from this node
        :param query_attr: scope being queried with length k
        :param query_shape: the input query is flattened to n,k from query_shape _,k
        :return: probability of query, of shape n,
        """
        assert len(query) == 2, "incorrect query parser"
        if node is None:
            node = self.model
        if query_attr is None:
            query_attr = node.scope
        scope = node.scope
        condition = [item for item in node.range] if node.range is not None else []
        assert query[0].shape[-1] == len(scope)+len(condition), "query length mismatch"
        assert set(query_attr) == set(scope+condition), "incorrect query_attr"

        #get the first factorize node in this branch
        first = None
        exist_fact = False
        for fact in get_nodes_by_type(node, Factorize):
            exist_fact = True
            if fact.id not in calculated:
                if first is None or fact.id < first.id:
                    first = fact
        if not exist_fact:
            #This node does not have factorize node children
            prob = self._probability_left_most(query, node, query_attr)
            return prob
        elif first is None:
            #We have evaluated all factorize node
            prob = self._spn_probability(query, node, query_attr, calculated)
            return prob
        else:
            prob = self.eval_fact_node(query, first, query_attr, calculated)
            calculated[first.id] = prob
            if node.id == first.id:
                #factorize node is the root node in the current branch
                return prob
            else:
                return self.probability(query, node, query_attr, calculated)


    def eval_fact_node(self, query, node, query_attr, calculated):
        """
        input query shape: n,k
        output probability: n,
        """
        right_branch = node.children[-1]
        scope = right_branch.scope
        condition = [item for item in right_branch.range]
        assert len(set(scope).intersection(set(condition))) == 0, "some scope conditioned on itself"

        scope_idx = [query_attr.index(item) for item in scope]
        condition_idx = [query_attr.index(item) for item in condition]
        if type(query) != tuple:
            scope_query = query[:, scope_idx]
            condition_query = query[:, condition_idx]
        else:
            scope_query = (query[0][:, scope_idx], query[1][:, scope_idx])
            condition_query = (query[0][:, condition_idx], query[1][:, condition_idx])

        scope_prob = self._leave_prob(scope_query, node.id, scope)
        
        new_query = self.get_overlap(self.leaves_range[node.id], condition_query)
        
        condition_prob = self.probability(new_query, node.children[0], condition, calculated)
        condition_prob = condition_prob.reshape((-1, len(self.leaves[node.id])))
        
        prob = np.sum(np.multiply(condition_prob, scope_prob), axis=1)
        prob[prob > 1] = 1
        return prob

    def likelihood(self, data, node=None, attr=None, calculated=dict(), log=False):
        """
        compute the likelihood of data in a top-down and bottom up manner.
        """
        if node is None:
            node = self.model
        if attr is None:
            attr = node.scope

        scope = node.scope
        condition = [item for item in node.range] if node.range is not None else []
        assert data.shape[-1] == len(scope) + len(condition), "query length mismatch"
        assert set(attr) == set(scope + condition), "incorrect query_attr"

        # get the first factorize node in this branch
        first = None
        exist_fact = False
        for fact in get_nodes_by_type(node, Factorize):
            exist_fact = True
            if fact.id not in calculated:
                if first is None or fact.id < first.id:
                    first = fact
        if not exist_fact:
            # This node does not have factorize node children
            ll = self._likelihood_left_most(data, node, attr, log)
            return ll
        elif first is None:
            # We have evaluated all factorize node
            ll = self._spn_likelihood(data, node, attr, calculated, log)
            return ll
        else:
            ll = self.eval_fact_node_likelihood(data, first, attr, calculated, log)
            calculated[first.id] = ll
            if node.id == first.id:
                # factorize node is the root node in the current branch
                return ll
            else:
                return self.likelihood(data, node, attr, calculated, log)

    def _likelihood_left_most(self, data, node, attr, log=False):
        """
            calculate the probability on spn without factorized node
        """
        nodes = get_topological_order(node)
        all_results = {}

        for n in nodes:
            if isinstance(n, Leaf):
                result = n.likelihood(data, attr, log)
            else:
                tmp_children_list = []
                for i in range(len(n.children)):
                    ci = n.children[i]
                    tmp_children_list.append(all_results[ci])
                if isinstance(n, Sum):
                    if log:
                        result = sum_log_likelihood(n, tmp_children_list)
                    else:
                        result = sum_likelihood(n, tmp_children_list)
                elif isinstance(n, Product):
                    if log:
                        result = prod_log_likelihood(n, tmp_children_list)
                    else:
                        result = prod_likelihood(n, tmp_children_list)
                else:
                    assert not isinstance(n, Factorize), "Factorize node should be eliminated"

            all_results[n] = result

        return all_results[node]

    def _spn_likelihood(self, data, node, attr, calculated, log):
        """
            Calculate the probability of a branch with all factorized node evaluated
            Node must contain a factorize node as children in this case.
        """
        assert data.shape[-1] == len(attr)
        if isinstance(node, Leaf):
            res = node.likelihood(data, attr, log)
            return res
        elif isinstance(node, Factorize):
            return calculated[node.id]

        child_res = []
        for child in node.children:
            child_res.append(self._spn_likelihood(data, child, attr, calculated, log))
        if isinstance(node, Sum):
            if log:
                return sum_log_likelihood(node, child_res)
            return sum_likelihood(node, child_res)
        else:
            if log:
                return prod_log_likelihood(node, child_res)
            return prod_likelihood(node, child_res)

    def right_branch_likelihood(self, scope_data, condition_data, node, log):
        """
        Evaluate the data likelihood in a top-down fashion.
        """
        right_branch = node.children[-1]
        scope = right_branch.scope
        condition = [item for item in right_branch.range]
        ranges = self.leaves_range[node.id]
        leaves = self.leaves[node.id]

        scope_ll = np.zeros(len(scope_data))
        al, ar = ranges
        (n, k) = al.shape
        assert k == len(condition)
        total_len = 0
        for i in range(n):
            left_res = np.maximum(condition_data, al[i, :])
            right_res = np.minimum(condition_data, ar[i, :])
            illegal_idx = np.unique(np.where(left_res > right_res)[0])
            legal_idx = np.setdiff1d(np.arange(len(scope_data)), illegal_idx)
            legal = scope_data[legal_idx]
            total_len += len(legal)
            scope_ll[legal_idx] = leaves[i].likelihood(legal, scope, log)

        assert total_len == len(scope_ll)
        return scope_ll

    def eval_fact_node_likelihood(self, data, node, attr, calculated, log):
        """
        input query shape: n,k
        output probability: n,
        """
        right_branch = node.children[-1]
        scope = right_branch.scope
        condition = [item for item in right_branch.range]
        assert len(set(scope).intersection(set(condition))) == 0, "some scope conditioned on itself"

        scope_idx = [attr.index(item) for item in scope]
        condition_idx = [attr.index(item) for item in condition]
        scope_data = data[:, scope_idx]

        condition_data = data[:, condition_idx]
        condition_ll = self.likelihood(condition_data, node.children[0], condition, calculated, log)

        scope_ll = self.right_branch_likelihood(scope_data, condition_data, node, log)

        if log:
            return condition_ll + scope_ll
        return condition_ll * scope_ll
