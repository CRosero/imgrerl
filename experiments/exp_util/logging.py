
def get_log_lines(iteration, log_dict):
    lines = []
    lines.append("---- Iteration: {:5d} ----".format(iteration))
    prefix = None
    current_line = None
    for k, v in log_dict.items():
        try:
            cur_prefix, cur_key = k.split("/")
            if prefix is None or prefix != cur_prefix:
                if current_line is not None:
                    lines.append(current_line[:-2])
                prefix = cur_prefix
                current_line = "{}: {}: {:.5f}, ".format(prefix, cur_key, v)
            else:
                current_line += "{}: {:.5f}, ".format(cur_key, v)
        except ValueError:
            pass
    lines.append(current_line[:-2])
    lines.append("---------------------------")
    return lines


def print_log(iteration, log_dict):
    lines = get_log_lines(iteration=iteration, log_dict=log_dict)
    for l in lines:
        print(l)
