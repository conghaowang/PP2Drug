import importlib
import torch
# from data_processing.paired_data import PharmacophoreDataset, CombinedGraphDataset
from torch_geometric.loader import DataLoader

def load_dataset(cls_name, root, split):
    path = f'data_processing.paired_data'
    parent_module = importlib.import_module(path)
    dataset = getattr(parent_module, cls_name)(root=root, split=split)
    return dataset

def load_data(cls_name, root, split='train', batch_size=32, num_workers=0):
    # dataset = PharmacophoreDataset(root=root, split=split)
    dataset = load_dataset(cls_name, root, split)
    if split == 'train':
        return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=True):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)