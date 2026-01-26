import torch
from tqdm import tqdm
from logger import get_model_save_path


def get_penalty(penalty, poly_activation):
    """
    Compute regularization penalty for polynomial activations.

    Args:
        penalty (float): Penalty coefficient.
        poly_activation (PolyActivation): Activation module containing buffered activations.

    Returns:
        torch.Tensor: Scalar tensor with penalty value.
    """
    activations = poly_activation.buff
    clip = poly_activation.clip
    penalties = []
    for act_poly in activations:
        diff_clip = (act_poly - torch.clamp(act_poly, -clip, clip)).flatten()
        l2_clip_norm = torch.linalg.norm(diff_clip, ord=2)
        add_penalty = penalty * l2_clip_norm
        penalties.append(add_penalty)
    
    total_penalty = torch.mean(torch.stack(penalties))
    poly_activation.reset()
    return total_penalty


def train(epoch, model, trainloader, criterion, optimizer, penalty, poly_activation):
    """
    Perform one epoch of training.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): Model being trained.
        trainloader (torch.utils.data.DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        penalty (float): Penalty coefficient for polynomial activation.
        poly_activation (PolyActivation or None): Polynomial activation module.

    Returns:
        tuple: (float, float): Average training loss and training accuracy percentage.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    warmup = epoch <= 4

    progress_bar = tqdm(trainloader, desc=f"Training Epoch {epoch}")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if poly_activation:
            if not warmup:
                add_penalty = get_penalty(penalty, poly_activation)
            else:
                if epoch == 1:
                    loss = 0
                    add_penalty = get_penalty(penalty / 100, poly_activation)
                elif epoch == 2:
                    add_penalty = get_penalty(penalty / 50, poly_activation)
                elif epoch == 3:
                    add_penalty = get_penalty(penalty / 10, poly_activation)
                else:
                    add_penalty = get_penalty(penalty / 5, poly_activation)
            loss = loss + add_penalty
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def validate(epoch, model, validloader, best_acc, model_path, args):
    """
    Validate model performance and save best checkpoint.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): Model to validate.
        validloader (torch.utils.data.DataLoader): Validation data loader.
        best_acc (float): Best validation accuracy so far.
        model_path (str): Directory where model checkpoints are saved.
        args (argparse.Namespace): Experiment configuration with
        seed (int): Seed that selects which checkpoint to load.

    Returns:
        tuple: (float, float): Current validation accuracy and updated best accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    progress_bar = tqdm(validloader, desc=f"Validating Epoch {epoch}")
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar.set_postfix(acc=f"{100. * correct / total:.4f}")
    valid_acc = 100. * correct / total

    if valid_acc > best_acc:
        best_acc = valid_acc
        model_save_path = get_model_save_path(model_path, args.seed)
        torch.save(model.state_dict(), model_save_path)
    return valid_acc, best_acc


def test(model, testloader):
    """
    Evaluate model on the test dataset.

    Args:
        model (torch.nn.Module): Trained model.
        testloader (torch.utils.data.DataLoader): Test data loader.

    Returns:
        float: Test accuracy percentage.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_acc = 100. * correct / total
    return test_acc
