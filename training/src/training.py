import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from arguments import parse_arguments
from dataloader import get_data_loaders
from logger import setup_paths, setup_logging, log_epoch_info, log_test_info, get_model_save_path
from models import get_model, replace_model
from noise_hooks import add_activation_noise
from poly_activation import PolyActivation
from trainer import train, validate, test
from utils import set_seed, save_dataset, load_dataset, DATASET_DIR, ARTIFACTS_DIR


def main():
    """
    Run the training and evaluation pipeline.
    """
    args = parse_arguments()
    set_seed(args.seed)

    log_path, model_path = setup_paths(args)
    setup_logging(log_path, args.seed)

    trainloader, validloader, testloader = get_data_loaders(args)

    dataset_path = os.path.join(ARTIFACTS_DIR, DATASET_DIR, args.dataset)
    os.makedirs(dataset_path, exist_ok=True)
    testdataset_path = os.path.join(dataset_path, f'testdataset_seed_{args.seed}.pkl')
    if not os.path.exists(testdataset_path):
        save_dataset(testloader, testdataset_path)

    criterion = nn.CrossEntropyLoss()
    model = get_model(args).cuda()

    if args.sigma > 0:
        model = add_activation_noise(model, args.sigma)
    
    poly_activation = None
    if "poly" in args.model:
        poly_activation = PolyActivation(args.degree, args.clip, args.pbit)
        replace_model(model, poly_activation)

    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    warmup_lr_scheduler = LinearLR(optimizer, total_iters=5, start_factor=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - 5, eta_min=0)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, scheduler], milestones=[5])

    best_acc = 0.0
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train(epoch, model, trainloader, criterion, optimizer, args.penalty, poly_activation)
        valid_acc, best_acc = validate(epoch, model, validloader, best_acc, model_path, args)
        lr_scheduler.step()
        log_epoch_info(epoch, train_loss, train_acc, valid_acc)

    model_save_path = get_model_save_path(model_path, args.seed)
    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    test_dataset = load_dataset(testdataset_path)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    test_acc = test(model, testloader)
    log_test_info(test_acc)


if __name__ == "__main__":
    main()
