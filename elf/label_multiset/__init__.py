"""Implementation of the label multiset format used by paintera.
"""

from .create import create_multiset_from_labels, downsample_multiset, merge_multisets
from .label_multiset import LabelMultiset
from .serialize import serialize_multiset, deserialize_multiset, deserialize_labels
