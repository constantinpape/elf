"""Wrapper for large image data to implement lazy data processing.
"""

from .base import SimpleTransformationWrapper, SimpleTransformationWrapperWithHalo, TransformationWrapper
from .generic import NormalizeWrapper, ThresholdWrapper, RoiWrapper
