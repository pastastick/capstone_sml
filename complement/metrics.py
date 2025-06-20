from sklearn import metrics
from skimage import measure
import cv2
import numpy as np
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_recall_curve, accuracy_score

def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights, pos_label=None):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_idx = np.argmax(f1_scores) 
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    
    y_pred = (anomaly_prediction_weights >= best_threshold).astype(int)
    if pos_label is not None and pos_label != 1:
        y_pred = np.where(y_pred == 1, pos_label, anomaly_ground_truth_labels.min())
    
    best_accuracy = accuracy_score(anomaly_ground_truth_labels, y_pred)
    
    # print(best_threshold, best_precision, best_recall)

    return {"precision": best_precision,
            "recall": best_recall,
            "accuracy": best_accuracy,
            "f1_score": best_f1,
            "threshold": best_threshold,
            }

def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    ap = 0. if path == 'training' else metrics.average_precision_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

    return {"auroc": auroc, "ap": ap}


# def compute_pro(masks, amaps, num_th=200):
#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
#     binary_amaps = np.zeros_like(amaps, dtype=bool)

#     min_th = amaps.min()
#     max_th = amaps.max()
#     delta = (max_th - min_th) / num_th

#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     for th in np.arange(min_th, max_th, delta):
#         binary_amaps[amaps <= th] = 0
#         binary_amaps[amaps > th] = 1

#         pros = []
#         for binary_amap, mask in zip(binary_amaps, masks):
#             binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
#             for region in measure.regionprops(measure.label(mask)):
#                 axes0_ids = region.coords[:, 0]
#                 axes1_ids = region.coords[:, 1]
#                 tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
#                 pros.append(tp_pixels / region.area)

#         inverse_masks = 1 - masks
#         fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
#         fpr = fp_pixels / inverse_masks.sum()

#         df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, index=[0])])

#     df = df[df["fpr"] < 0.3]
#     df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

#     pro_auc = metrics.auc(df["fpr"], df["pro"])
#     return pro_auc

def compute_pro(masks, amaps, num_th=200):
    # Inisialisasi list untuk menyimpan data sementara
    data = []
    
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # Pastikan pros tidak kosong sebelum menghitung mean
        pro_mean = np.mean(pros) if pros else 0.0
        # Tambahkan data ke list
        data.append({"pro": pro_mean, "fpr": fpr, "threshold": th})

    # Buat DataFrame dari list data
    df = pd.DataFrame(data)

    # Filter berdasarkan fpr < 0.3
    df = df[df["fpr"] < 0.3]
    # Normalisasi fpr
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

    # Hitung AUC
    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc