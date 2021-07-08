def convert_to_scope_domain(scope, condition):
    """
    First create a list that merge scope and condition list in sorted order
    """
    scope_range = []  # put scope and condition together in sorted order
    s_i = 0
    c_i = 0
    # remember the location
    scope_loc = []
    condition_loc = []
    while len(scope_range) < len(scope) + len(condition):
        if s_i >= len(scope):
            scope_range.append(condition[c_i])
            condition_loc.append(s_i + c_i)
            c_i += 1
        elif c_i >= len(condition):
            scope_range.append(scope[s_i])
            scope_loc.append(s_i + c_i)
            s_i += 1
        else:
            if scope[s_i] < condition[c_i]:
                scope_range.append(scope[s_i])
                scope_loc.append(s_i + c_i)
                s_i += 1
            else:
                scope_range.append(condition[c_i])
                condition_loc.append(s_i + c_i)
                c_i += 1
    return scope_range, scope_loc, condition_loc


def get_matached_domain(idx, scope, condition):
    assert len(idx) == (len(scope)+len(condition))
    scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
    scope_idx = []
    rm_scope = []
    new_scope = []
    condition_idx = []
    new_condition = []
    for i in range(len(idx)):
        if idx[i]:
            if i in scope_loc:
                scope_idx.append(i)
                rm_scope.append(scope_loc.index(i))
            else:
                condition_idx.append(i)
        else:
            if i in scope_loc:
                new_scope.append(scope_loc.index(i))
            else:
                new_condition.append(condition_loc.index(i))

    return scope_idx, rm_scope, new_scope, condition_idx, new_condition
