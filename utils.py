from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, precision_recall_curve, auc
import numpy
import numpy as np


def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def get_mf1_auc_aucpr( labels, probs, train_mask, val_mask, test_mask):
    train_mask = train_mask
    val_mask = val_mask
    test_mask = test_mask

    train_mf1, _ = get_best_f1(labels[train_mask], probs[train_mask])
    vmf1, thres = get_best_f1(labels[val_mask], probs[val_mask])
    preds = numpy.zeros_like(labels)
    preds[probs[:, 1] > thres] = 1
    trec = recall_score(labels[test_mask], preds[test_mask])
    tpre = precision_score(labels[test_mask], preds[test_mask])
    tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
    v_mif1 = f1_score(labels[val_mask], preds[val_mask], average='micro')
    t_maf1 = f1_score(labels[test_mask], preds[test_mask], average='micro')
    tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())
    precision, recall, _ = precision_recall_curve(labels[test_mask], probs[test_mask][:, 1].detach())
    tauc_pr = auc(recall, precision)
    vprecision, vrecall, _ = precision_recall_curve(labels[val_mask], probs[val_mask][:, 1].detach())
    vauc_pr = auc(vrecall, vprecision)
    vauc = roc_auc_score(labels[val_mask], probs[val_mask][:, 1].detach().numpy())

    return (vmf1, vauc, vauc_pr), (tmf1, tauc, tauc_pr), trec, tpre, train_mf1,v_mif1,t_maf1