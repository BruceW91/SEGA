import random
from typing import Any, Optional, Union
from typing import List
from typing import Dict

import fsspec
import numpy as np
import torch
from torch import Tensor

from torch.utils.data import default_collate


# from llava.dataset.zb_refine_noise_utils.metric import compute_validity

DUMMY_LAYOUT = {
    "label": 0,
    "center_x": 0.5,
    "center_y": 0.5,
    "width": 0.05,
    "height": 0.05,
}
DUMMY_LAYOUT = {k: [v] for k, v in DUMMY_LAYOUT.items()}

def collate_fn(
    batch,
    max_seq_length=20,
    validity_check = None,
):
    """
    Custom function to merge varying-length inputs into a single batch.
    For padding, we used the values defined in pad() function above.
    """
    assert (
        validity_check is None
    ), f"validity_check function is {validity_check}"

    B = len(batch)

    # delete special column used to pass the name of transforms
    for i in range(B):
        if "transforms" in batch[i]:
            del batch[i]["transforms"]

    # check if all the elements in a batch have the length > 0
    # (sometimes, generated layouts are empty)
    total_elems = []
    for i in range(B):
        n = len(batch[i]["label"])
        if n == 0:
            # add dummy element to continue evaluation
            for k in DUMMY_LAYOUT:
                batch[i][k] = DUMMY_LAYOUT[k]
            n = 1
        total_elems.append(n)

    output = {}

    for key in batch[0].keys():

        main_data = batch[0][key]
        if not isinstance(main_data, list) or len(main_data) == 0:
            continue

        # number of elements in a layout varies, so we need padding
        if isinstance(main_data[0], int):
            pad_value = 0
        elif isinstance(main_data[0], float):
            pad_value = 0.0
        else:
            # assume this type of data works without padding
            batch[i][key] = torch.tensor(batch[i][key])
            continue

        for i in range(B):
            batch[i][key] = torch.tensor(
                batch[i][key] + [pad_value] * (max_seq_length - total_elems[i])
            )

    for i in range(B):
        n = total_elems[i]
        batch[i]["mask"] = torch.BoolTensor([True] * n + [False] * (max_seq_length - n))

    if validity_check is not None:
        batch, _ = validity_check(batch)

    output = {**output, **default_collate(batch)}
    return output


def is_run_on_local(dir_name: str) -> bool:
    """
    Check if script is run local machine or not by the given directory / file name.
    """
    fs, _ = fsspec.core.url_to_fs(dir_name)
    return isinstance(fs, fsspec.implementations.local.LocalFileSystem)


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x: List[Union[int, float]]) -> List[int]:
    assert isinstance(x, list) and isinstance(x[0], (int, float))
    return sorted(range(len(x)), key=x.__getitem__)


def is_dict_of_list(x: Any) -> bool:
    if isinstance(x, dict):
        return all(isinstance(v, list) for v in x.values())
    else:
        return False


def dict_of_list_to_list_of_dict(dl: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


def is_list_of_dict(x: Any) -> bool:
    if isinstance(x, list):
        return all(isinstance(d, dict) for d in x)
    else:
        return False


def list_of_dict_to_dict_of_list(ld: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def pad(data: List[Any], max_seq_length: int) -> List[Any]:
    assert len(data) > 0
    value = data[0]
    if isinstance(value, bool):
        pad_value = False
    elif isinstance(value, int):
        pad_value = 0
    elif isinstance(value, float):
        pad_value = 0.0
    else:
        raise NotImplementedError

    n = len(data)
    assert n <= max_seq_length
    return data + [pad_value] * (max_seq_length - n)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def convert_xywh_to_ltrb(
    bbox: Union[Tensor, np.ndarray, List[float]]
) -> Union[List[Tensor], List[np.ndarray], List[float]]:
    assert len(bbox) == 4
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def batch_shuffle_index(
    batch_size: int,
    feature_length: int,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.LongTensor:
    """
    Note: masked part may be shuffled because of
    unpredictable behaviour of sorting [inf, ..., inf]
    """
    if mask:
        assert list(mask.size()) == [batch_size, feature_length]
    scores = torch.rand((batch_size, feature_length))
    if mask:
        scores[~mask] = float("Inf")
    indices: torch.LongTensor = torch.sort(scores, dim=1)[1]
    return indices
