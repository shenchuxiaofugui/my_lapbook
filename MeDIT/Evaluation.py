import numpy as np
from scipy.ndimage import label

def IoU(predict_mask, label_mask, constrain_2d=True):
    predict_image, predict_label_value = label(predict_mask)
    label_image, label_value = label(label_mask)

    # 确保图像中只有一个区域
    assert(predict_label_value == 1)
    assert(label_value == 1)

    if not constrain_2d:
        return (predict_mask * label_mask).astype(float).sum() / (predict_mask + label_mask >= 1).astype(float).sum()
    else:
        slice_index = np.where(np.sum(predict_mask * label_mask, axis=(0, 1)))
        if len(slice_index) == 0:
            return 0
        else:
            sub_predict, sub_label = predict_mask[..., slice_index], label_mask[..., slice_index]
            return (sub_predict * sub_label).astype(float).sum() / (sub_predict + sub_label >= 1).astype(float).sum()

def LabelIoU(predict_mask, label_mask, constrain_2d=True):
    predict_image, predict_label_value = label(predict_mask)
    label_image, label_value = label(label_mask)

    # 确保图像中只有一个区域
    assert (predict_label_value == 1)
    assert (label_value == 1)

    if not constrain_2d:
        return (predict_mask * label_mask).astype(float).sum() / label_mask.sum()
    else:
        slice_index = np.where(np.sum(predict_mask * label_mask, axis=(0, 1)))
        if len(slice_index) == 0:
            return 0
        else:
            sub_predict, sub_label = predict_mask[..., slice_index], label_mask[..., slice_index]
            return (sub_predict * sub_label).astype(float).sum() / sub_label.sum()