import importlib
import torch
# from data_processing.paired_data import PharmacophoreDataset, CombinedGraphDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

def load_dataset(cls_name, root, split):
    if cls_name == 'QM9Dataset':
        path = f'data_processing.qm9_data'
    else:
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
    

def load_qm9_data(cls_name='QM9Dataset', root='../data/qm9', split='train', batch_size=32, num_workers=0):
    dataset = load_dataset(cls_name, root, split)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = int(0.1 * total_size)  # 10% for validation
    test_size = total_size - train_size - val_size  # Remaining for testing

    generator = torch.Generator().manual_seed(2024)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    if split == 'train':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataset, val_dataset, train_loader, val_loader
    elif split == 'test':
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_dataset, test_loader
    else:
        return ValueError(f"Invalid split: {split}")
    

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