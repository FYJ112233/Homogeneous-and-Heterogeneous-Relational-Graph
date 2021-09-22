import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


class CenterTripletLoss(nn.Module):
    
    def __init__(self, batch_size, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):

        label_uni = labels.unique()
        targets = torch.cat([label_uni,label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num*2, 0)
        center = []
        for i in range(label_num*2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)
        
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):

    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class CMCC(nn.Module):
    def __init__(self, batch_size, margin=0.3):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    def forward(self, inputs, targets):

        feat_V, feat_T = torch.chunk(inputs, 2, dim=0)
        c = feat_V.T @ feat_T  # empirical cross-correlation matrix
        n = inputs.size(0)
        c.div_(n) 
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = (on_diag + 0.050 * off_diag) / feat_V.size（0）
        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def pdist_torch(emb1, emb2):

    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):

    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def normalize(x, axis=-1):

    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def batch_euclidean_dist(x, y):

    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    N, m, d = x.size()
    N, n, d = y.size()

    # shape [N, m, n]
    xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
    yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
    dist = xx + yy
    dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def shortest_dist(dist_mat):

    m, n = dist_mat.size()[:2]
    # Just offering some reference for accessing intermediate distance.
    dist = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:
                dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
    dist = dist[-1][-1]
    return dist


def local_dist(x, y):

    M, m, d = x.size()
    N, n, d = y.size()
    x = x.contiguous().view(M * m, d)
    y = y.contiguous().view(N * n, d)
    # shape [M * m, N * n]
    dist_mat = euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
    # shape [M, N]
    dist_mat = shortest_dist(dist_mat)
    return dist_mat


def batch_local_dist(x, y):

    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    # shape [N, m, n]
    dist_mat = batch_euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [N]
    dist = shortest_dist(dist_mat.permute(1, 2, 0))
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):


    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def global_loss(tri_loss, global_feat, labels, normalize_feature=True):

    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels, return_inds=True)
    loss = tri_loss(dist_ap, dist_an)
    return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat


def local_loss(
        tri_loss,
        local_feat,
        p_inds=None,
        n_inds=None,
        labels=None,
        normalize_feature=True):
    if normalize_feature:
        local_feat = normalize(local_feat, axis=-1)
    if p_inds is None or n_inds is None:
        dist_mat = local_dist(local_feat, local_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels, return_inds=False)
        loss = tri_loss(dist_ap, dist_an)
        return loss, dist_ap, dist_an, dist_mat
    else:
        dist_ap = batch_local_dist(local_feat, local_feat[p_inds])
        dist_an = batch_local_dist(local_feat, local_feat[n_inds])
        loss = tri_loss(dist_ap, dist_an)
        return loss, dist_ap, dist_an




def normalize(x, axis=-1):

  x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
  return x


class HGAM_global_loss_idx(nn.Module):

    def __init__(self, batch_size, margin=0.3):
        super(global_loss_idx, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, global_feat, labels):
        global_feat = normalize(global_feat, axis=-1)
        inputs = global_feat
        n = inputs.size(0)
        dist_mat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_mat = dist_mat + dist_mat.t()
        dist_mat.addmm_(1, -2, inputs, inputs.t())
        dist_mat = dist_mat.clamp(min=1e-12).sqrt()

        N = dist_mat.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        dist_ap, relative_p_inds = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        dist_an, relative_n_inds = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))

        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)


        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)
        global_feat = global_feat.chunk(label_num * 2, 0)
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(global_feat[i], dim=0, keepdim=True))

        inputs = torch.cat(center)

        n = inputs.size(0)

        dist_c = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_c = dist_c + dist_c.t()
        dist_c.addmm_(1, -2, inputs, inputs.t())
        dist_c = dist_c.clamp(min=1e-12).sqrt()  

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap_c, dist_an_c = [], []
        for i in range(n):
            dist_ap_c.append(dist_c[i][mask[i]].max().unsqueeze(0))
            dist_an_c.append(dist_c[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap_c = torch.cat(dist_ap_c)
        dist_an_c = torch.cat(dist_an_c)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an_c)
        loss = self.ranking_loss(dist_an_c, dist_ap_c, y)

        return loss, p_inds, n_inds, dist_ap, dist_an




class HGAM_local_loss_idx(nn.Module):

    def __init__(self, batch_size, margin=0.3):
        super(local_loss_idx, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, local_feat, p_inds, n_inds, labels):
        local_feat = normalize(local_feat, axis=-1)

        dist_ap = batch_local_dist(local_feat, local_feat[p_inds.long()])
        dist_an = batch_local_dist(local_feat, local_feat[n_inds.long()])



        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss, dist_ap, dist_an

def local_dist(x, y):
  M, m, d = x.size()
  N, n, d = y.size()
  x = x.contiguous().view(M * m, d)
  y = y.contiguous().view(N * n, d)
  # shape [M * m, N * n]
  dist_mat = euclidean_dist(x, y)
  dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
  # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
  dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
  # shape [M, N]
  dist_mat = shortest_dist(dist_mat)
  return dist_mat

def shortest_dist(dist_mat):

  m, n = dist_mat.size()[:2]
  dist = [[0 for _ in range(n)] for _ in range(m)]
  for i in range(m):
    for j in range(n):
      if (i == 0) and (j == 0):
        dist[i][j] = dist_mat[i, j]
      elif (i == 0) and (j > 0):
        dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
      elif (i > 0) and (j == 0):
        dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
      else:
        dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
  dist = dist[-1][-1]
  return dist

def hard_example_mining(dist_mat, labels, return_inds=False):


  assert len(dist_mat.size()) == 2
  assert dist_mat.size(0) == dist_mat.size(1)
  N = dist_mat.size(0)

  # shape [N, N]
  is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
  is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

  dist_ap, relative_p_inds = torch.max(
    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
  dist_an, relative_n_inds = torch.min(
    dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
  # shape [N]
  dist_ap = dist_ap.squeeze(1)
  dist_an = dist_an.squeeze(1)

  if return_inds:
    # shape [N, N]
    ind = (labels.new().resize_as_(labels)
           .copy_(torch.arange(0, N).long())
           .unsqueeze( 0).expand(N, N))
    # shape [N, 1]
    p_inds = torch.gather(
      ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    n_inds = torch.gather(
      ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    # shape [N]
    p_inds = p_inds.squeeze(1)
    n_inds = n_inds.squeeze(1)
    return dist_ap, dist_an, p_inds, n_inds

  return dist_ap, dist_an
