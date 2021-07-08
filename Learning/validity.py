from Structure.nodes import Sum, Product, Factorize, get_nodes_by_type
import logging

logger = logging.getLogger(__name__)


def is_consistent(node):
    """
    all children of a product node have different scope and the parent has no condition
    """

    assert node is not None

    for prod_node in reversed(get_nodes_by_type(node, Product)):
        nscope = set(prod_node.scope)

        if len(prod_node.children) == 0:
            return False, "Product node %s has no children" % (prod_node.id)

        #if len(prod_node.condition) != 0:
            #logger.warning(f"Product node {prod_node.id} has condition {prod_node.condition}")
            #return (False, "Product node %s has condition" % (prod_node.id))

        allchildscope = set()
        sum_features = 0
        for child in prod_node.children:
            sum_features += len(child.scope)
            allchildscope.update(child.scope)

        if allchildscope != nscope or sum_features != len(allchildscope):
            print(allchildscope, nscope, sum_features, len(allchildscope))
            return (False, "children of (prod) node %s do not have exclusive scope" % (prod_node.id))

    return True, None

def check_factorize_node(node):
    """ Check:
    1. All children of a product node have different scope
    2. The condition of one child must be the scope + condition of previous child
    3. The range of all children must be contained in the range of itself.
    """
    allchildscope = set()
    for fact_node in reversed(get_nodes_by_type(node, Factorize)):
        nscope = set(fact_node.scope)

        if len(fact_node.children) != 2:
            return False, "Fact node %s does not have exactly two children" % (fact_node.id)

        allchildscope.clear()
        sum_features = 0
        prev_child = None
        for child in fact_node.children:
            sum_features += len(child.scope)
            allchildscope.update(child.scope)
            if prev_child is None:
                if len(child.condition) != 0:
                    return (False, "children of (Fact) node %s has condition" % (fact_node.id))
            else:
                if not set(child.condition).issubset(set(prev_child.condition+prev_child.scope)):
                    return (False, "children of (Fact) node %s has incorrect condition" % (fact_node.id))
                if child.range is None or len(child.range) == 0:
                    return (False, "children of (Fact) node %s has no range" % (fact_node.id))
            prev_child = child

        if allchildscope != nscope or sum_features != len(allchildscope):
            print(allchildscope, nscope, sum_features, len(allchildscope))
            return (False, "children of (Fact) node %s do not have exclusive scope" % (fact_node.id))

    return True, None

def is_complete(node):
    """
    1. All children of a sum node have same scope as the parent
    2. The condition of all children must be a subset of parent's condition
    3. The range of all children must be a subset of parent's range
    """

    assert node is not None

    for sum_node in reversed(get_nodes_by_type(node, Sum)):
        nscope = set(sum_node.scope)
        ncondition = set(sum_node.condition)

        if len(sum_node.children) == 0:
            return False, "Sum node %s has no children" % (sum_node.id)

        for child in sum_node.children:
            if nscope != set(child.scope):
                return (False, "children of (sum) node %s do not have the same scope as parent" % (sum_node.id))
            if len(ncondition) != 0 and not set(child.condition).issubset(ncondition):
                print(child.scope, child.condition)
                print(nscope, ncondition)
                return (False, "children of (sum) node %s 's conditon is not subset of parent's" % (sum_node.id))

    return True, None

def is_valid_range(parent, children):
    """Check if parent range covers children's range and children's range are disjoint"""
    children_range = dict()
    for child in children:
        if child.range is not None and len(child.range) != 0:
            for cond in child.range:
                if cond not in children_range:
                    children_range[cond] = child.range
                else:
                    if children_range[cond] != child.range:
                        continue
    return True

def is_valid(node, check_ids=True):

    if check_ids:
        val, err = has_valid_ids(node)
        if not val:
            return val, err

    for n in get_nodes_by_type(node):
        if len(n.scope) == 0:
            return False, "node %s has no scope" % (n.id)

        is_sum = isinstance(n, Sum)
        is_prod = isinstance(n, Product)
        is_factorize = isinstance(n, Factorize)

        if len(set(n.condition).intersection(set(n.scope))) != 0:
            return False, "node %s has same attribute in both condition and range" % (n.id)

        if is_sum:
            if len(n.children) != len(n.weights):
                return False, "node %s has different children/weights" % (n.id)

        if is_sum or is_prod or is_factorize:
            if len(n.children) == 0:
                return False, "node %s has no children" % (n.id)


    a, err = is_consistent(node)
    if not a:
        return a, err

    b, err = is_complete(node)
    if not b:
        return b, err

    c, err = check_factorize_node(node)
    if not c:
        return c, err

    return True, None


def has_valid_ids(node):
    ids = set()
    all_nodes = get_nodes_by_type(node)
    for n in all_nodes:
        ids.add(n.id)

    if len(ids) != len(all_nodes):
        return False, "Nodes are missing ids or there are repeated ids"

    if min(ids) != 0:
        return False, "Node ids not starting at 0"

    if max(ids) != len(ids) - 1:
        return False, "Node ids not consecutive"

    return True, None