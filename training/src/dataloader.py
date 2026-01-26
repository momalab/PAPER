from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from utils import dataset_map

"""
Constants and dataset mappings.

Attributes
    DATASET_TRANSFORMS: A mapping from dataset names to torchvision transforms specifying how images are preprocessed during training and evaluation.
"""

DATASET_TRANSFORMS = {
    "cifar10": {
        "train": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "test": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    },
    "cifar100": {
        "train": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "test": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    },
    "tiny": {
        "train": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        ]),
        "test": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        ])
    }
}


def get_transforms(dataset):
    """
    Return training and test transforms for the given dataset.

    Args:
        dataset (str): Dataset name.

    Returns:
        tuple: (torchvision.transforms.Compose, torchvision.transforms.Compose): Dataset transforms of train and test dataset.
    """
    if dataset not in DATASET_TRANSFORMS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASET_TRANSFORMS[dataset]["train"], DATASET_TRANSFORMS[dataset]["test"]


def get_datasets(dataset, transform_train, transform_test):
    """
    Load training and test datasets with given transforms.

    Args:
        dataset (str): Dataset name.
        transform_train (torchvision.transforms.Compose): Transform pipeline for training set.
        transform_test (torchvision.transforms.Compose): Transform pipeline for test set.

    Returns:
        tuple: (torch.utils.data.Dataset, torch.utils.data.Dataset): Train dataset and test dataset.
    """
    if dataset not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Tiny ImageNet-style datasets use separate train/test directories
    ds, root = dataset_map[dataset]
    if dataset in ["tiny"]:
        trainset = ds(root=f"{root}/train", transform=transform_train)
        testset = ds(root=f"{root}/val", transform=transform_test)
    else:
        trainset = ds(root=root, train=True, download=True, transform=transform_train)
        testset = ds(root=root, train=False, download=True, transform=transform_test)

    return trainset, testset


def get_data_loaders(args):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        args (argparse.Namespace): Experiment configuration with:
            dataset (str): Dataset name.
            batch_size_train (int): Training batch size.
            batch_size_test (int): Validation/testing batch size.
            validation_split (float): Fraction of test set used for validation.

    Returns:
        tuple: (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader): Dataloader for train, valid, and test dataset.
    """
    transform_train, transform_test = get_transforms(args.dataset)
    trainset, testset = get_datasets(args.dataset, transform_train, transform_test)

    valid_size = int(args.validation_split * len(testset))
    test_size = len(testset) - valid_size
    test_subset, valid_subset = random_split(testset, [test_size, valid_size])
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
    validloader = DataLoader(valid_subset, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_subset, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, validloader, testloader
