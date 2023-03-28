from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu==self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min]<self.thresh else sorteds[self.n_min]
            labels[picks>thresh] = self.ignore_lb
        ## TODO: here see if torch or numpy is faster
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss

class AutomaticWeightedLoss(nn.Module):
    
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class structure_loss(nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()
    
    def forward(self, pred, mask):
        weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
        wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
     
        pred  = torch.sigmoid(pred)
        inter = ((pred*mask)*weit).sum(dim=(2,3))
        union = ((pred+mask)*weit).sum(dim=(2,3))
        wiou  = 1-(inter+1)/(union-inter+1)
        return (wbce+wiou).mean()

# For this function, we use code from https://github.com/geovsion/EaNet
class ECELoss(nn.Module):
    def __init__(self, n_classes=19, alpha=1, radius=1, beta=0.5, ignore_lb=255, mode='ce', *args, **kwargs):
        super(ECELoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.n_classes = n_classes
        self.alpha = alpha
        self.radius = radius
        self.beta = beta
        if mode == 'ce':
            self.criteria = torch.nn.BCEWithLogitsLoss()
        elif mode == 'sce':
            self.criteria = OhemCELoss(0.7, 10000, 255)
        else:
            raise Exception('No %s loss, plase choose form ohem and ce' % mode)

        self.edge_criteria = EdgeLoss(self.n_classes, self.radius, self.alpha)

    def forward(self, logits, labels):
        if self.beta > 0:
            # return self.criteria(logits, labels)
            return self.criteria(logits, labels) + self.beta*self.edge_criteria(logits, labels)

class EdgeLoss(nn.Module):
    def __init__(self, n_classes=19, radius=1, alpha=0.01):
        super(EdgeLoss, self).__init__()
        self.n_classes = n_classes
        self.radius = radius
        self.alpha = alpha

    def forward(self, logits, label):
        prediction = torch.sigmoid(logits)
        ks = 2 * self.radius + 1
        filt1 = torch.ones(1, 1, ks, ks)
        filt1[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8
        filt1.requires_grad = False
        filt1 = filt1.cuda()

        lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=self.radius)
        lbedge = 1 - torch.eq(lbedge, 0).float()

        filt2 = torch.ones(self.n_classes, 1, ks, ks)
        filt2[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8
        filt2.requires_grad = False
        filt2 = filt2.cuda()
        prededge = F.conv2d(prediction.float(), filt2, bias=None,
                            stride=1, padding=self.radius, groups=self.n_classes)

        norm = torch.sum(torch.pow(prededge,2), 1).unsqueeze(1)
        prededge = norm/(norm + self.alpha)

        return BinaryDiceLoss()(prededge.float(),lbedge.float())

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        return loss.sum()

class JointClsLoss(nn.Module):

    def __init__(self, ignore_index=0, reduction='mean', bins=(5,5)):
        super(JointClsLoss, self).__init__()
        self.ignore_index = ignore_index
        self.cls_criterion = FocalLoss()
        self.bins = bins
        self.cls_weight = 1.0
        if not reduction:
            print("disabled the reduction.")

    def get_bin_label(self, label_onehot, bin_size):
        cls_percentage = F.adaptive_avg_pool2d(label_onehot, bin_size)
        cls_label = torch.where(cls_percentage>0, torch.ones_like(cls_percentage), torch.zeros_like(cls_percentage))

        return cls_label

    def forward(self, preds, target_onehot):
        cls_list = preds[1]
        cls_loss = 0
        for cls_pred, bin_size in zip(cls_list, self.bins):
            cls_gt = self.get_bin_label(target_onehot, bin_size)
            cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_gt) / len(self.bins)

        loss_dict = {'cls_loss':cls_loss}
        return loss_dict

class FocalLoss(nn.Module):
    ''' focal loss '''
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.crit = nn.BCELoss(reduction='none')

    def binary_focal_loss(self, input, target):
        ce_loss = self.crit(input, target)
        loss = ce_loss.mean()
        return loss
        
    def	forward(self, input, target):
        K = target.shape[1]
        total_loss = 0
        for i in range(K):
            total_loss += self.binary_focal_loss(input[:,i], target[:,i])

        return total_loss / K