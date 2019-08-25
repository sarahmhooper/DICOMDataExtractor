import logging

from itertools import compress
import pandas as pd

from emmental.metrics import METRICS
from emmental.utils.utils import array_to_numpy

logger = logging.getLogger(__name__)


class Scorer(object):
    """A class to score tasks

    :param metrics: a list of metric names which provides in emmental (e.g., accuracy)
    :type metrics: list
    :param customize_metric_funcs: a dict of customize metric where key is the metric
        name and value is the metric function which takes gold, preds, probs, uids as
        input
    :type customize_metric_funcs: dict
    """

    def __init__(self, metrics=[], customize_metric_funcs={}):
        self.metrics = dict()
        for metric in metrics:
            if metric not in METRICS:
                raise ValueError(f"Unrecognized metric: {metric}")
            self.metrics[metric] = METRICS[metric]

        self.metrics.update(customize_metric_funcs)
        
        self.frac_uids = []
        self.stroke_uids = []
        self.hem_uids = []
        
        for split in ["train","valid","test"]:
            data = pd.read_csv(f'data/csv/{split}.csv')[
                ["id", "abnormal_label", "weak_hem_label", "weak_stroke_label", "weak_frac_label"]
            ]
            frac_uids = data[data.weak_frac_label != 0]
            self.frac_uids += list(frac_uids['id'])
            stroke_uids = data[data.weak_stroke_label != 0]
            self.stroke_uids += list(stroke_uids['id'])
            hem_uids = data[data.weak_hem_label != 0]
            self.hem_uids += list(hem_uids['id'])
        
        self.frac_uids = set(self.frac_uids)
        self.stroke_uids = set(self.stroke_uids)
        self.hem_uids = set(self.hem_uids)

    def score(self, golds, preds, probs, uids=None):
        metric_dict = dict()

        for metric_name, metric in self.metrics.items():
                        
            if metric_name in ['frac_recall','hem_recall','stroke_recall']:
                if metric_name == 'frac_recall': slice_uids = self.frac_uids
                elif metric_name == 'hem_recall': slice_uids = self.hem_uids
                elif metric_name == 'stroke_recall': slice_uids = self.stroke_uids
                uid_inslice = [t in slice_uids for t in uids]
                uids = list(compress(uids, uid_inslice))
                golds = list(compress(golds, uid_inslice))
                preds = list(compress(preds, uid_inslice))
                probs = list(compress(probs, uid_inslice))

            # handle no examples
            if len(golds) == 0:
                metric_dict[metric_name] = float("nan")
                continue

            golds = array_to_numpy(golds)
            preds = array_to_numpy(preds)
            probs = array_to_numpy(probs)
              
            res = metric(golds, preds, probs, uids)
            if isinstance(res, dict):
                metric_dict.update(res)
            else:
                metric_dict[metric_name] = res

        return metric_dict
