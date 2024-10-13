import os


def gen_folder_name(
    directory, verbose=False, offset=0, ret_full=True, make_folder=True
):
    folders = []
    for f in os.listdir(directory):
        try:
            folders.append(int(f))
        except ValueError:
            if verbose:
                print(f'Ignoring non-integer name "{f}"')
    if len(folders) < 1:
        index = 0
    else:
        index = max(folders) + 1

    # Left pad the index with zeros (Ex: 0051 instead of 51)
    name = f"{index+offset:0>4d}"
    if make_folder:
        os.mkdir(os.path.join(directory, name))
    if ret_full:
        return os.path.join(directory, name)
    else:
        return name


def MAV(values, L=10):
    if len(values) < 1:
        return []
    # Moving average
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + (v - out[-1]) / L)
    return out


def set_req_grad(model, is_req=True):
    for param in model.parameters():
        param.requires_grad = is_req
