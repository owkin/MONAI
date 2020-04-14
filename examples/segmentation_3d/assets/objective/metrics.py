import torch
from monai.metrics import compute_meandice

import substratools as tools


class MonaiMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        metric_sum = 0.
        metric_count = 0
        with torch.no_grad():
            for (val_true, val_pred) in zip(y_true, y_pred):
                val_true, _ = val_true
                val_pred, _ = val_pred
                value = compute_meandice(y_pred=val_pred,
                                         y=val_true,
                                         include_background=True,
                                         to_onehot_y=False,
                                         add_sigmoid=True)
                metric_count += len(value)
                metric_sum += value.sum().item()
        metric = metric_sum / metric_count
        return metric


if __name__ == '__main__':
    tools.metrics.execute(MonaiMetrics())
