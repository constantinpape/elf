"""Tracking metrics implemented with [traccuracy](https://github.com/Janelia-Trackathon-2023/traccuracy).
"""
import traccuracy
from traccuracy import run_metrics
from traccuracy.matchers import CTCMatched
from traccuracy.metrics import CTCMetrics, DivisionMetrics


def build_gt_graph():
    pass


def compute_ctc_metric(lineage_graph, gt_graph):
    pass
