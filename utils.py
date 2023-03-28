import torch

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    num_epochs = num_epochs+5
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def eval_mae(y_pred, y):
    return torch.abs(y_pred - y).mean()

def eval_pr(y_pred, y, num):
    
    prec, recall = torch.zeros(num), torch.zeros(num)
    thlist       = torch.linspace(0, 1, num)
    if y_pred.is_cuda:
        thlist = thlist.to('cuda:0')
    for i in range(num):
        y_temp   = (y_pred >= thlist[i]).float()
        tp       = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall

def F_score(y_pred,y):
    beta = 0.3
    mean = torch.mean(y_pred)
    # y_pred[y_pred < mean*1.5] = 0
    # y_pred[y_pred >= mean*1.5] = 1

    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    prec, recall = eval_pr(y_pred, y, 255)
    score = (1 + beta) * prec * recall / (beta * prec + recall + 1e-10)
    score[score != score] = 0
    f_score = score.max().item()
    return f_score

def F_score1(y_pred,y):
    beta = 0.3
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    prec, recall = eval_pr(1-y_pred, 1-y, 255)
    score = (1 + beta) * prec * recall / (beta * prec + recall + 1e-10)
    score[score != score] = 0
    f_score = score.max().item()
    return f_score