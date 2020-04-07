import torch
import numpy as np

#adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py

class iouEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.batchCount = 0

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        if isinstance(predict, np.ndarray):
            predict = predict.flatten()
            gth = gth.flatten()
        elif isinstance(predict, torch.Tensor):
            predict = predict.cpu().numpy().flatten()
            gth = gth.cpu().numpy().flatten()

        epsilon = 0.00000001
        if self.batchCount == 0:
            self.hist = self.compute_hist(predict, gth)
        else:
            self.hist += self.compute_hist(predict, gth)
        hist = self.compute_hist(predict, gth)
        # hist(0) : TP + FN
        # hist(1) : TP + FP
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        mIou = np.nanmean(per_class_iu)

        self.overall_acc +=overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iu += per_class_iu
        self.mIOU += mIou
        self.batchCount += 1
        return hist

    def getMetric(self):
        epsilon = 0.00000001
        overall_acc = np.diag(self.hist).sum() / (self.hist.sum() + epsilon)
        per_class_acc = np.diag(self.hist) / (self.hist.sum(1) + epsilon)
        per_class_iu = np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist) + epsilon)
        mIOU = np.nanmean(per_class_iu)
        return overall_acc, per_class_acc, per_class_iu, mIOU
        
