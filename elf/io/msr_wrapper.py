import os
from collections.abc import Mapping, Sequence
from typing import Union

import numpy as np

from ..util import normalize_index, squeeze_singletons

try:
    from msr_reader import OBFFile
except ImportError:
    OBFFile = None


_MSR_READER_INSTALL_ERROR = "msr_reader is required for MSR images, but is not installed. Install it with `pip install msr-reader`."

StackIndexSelection = Union[int, Sequence[int]]
StackNameSelection = Union[str, Sequence[str]]
StackSelection = Union[int, str, Sequence[Union[int, str]]]
PathLike = Union[os.PathLike, str]


def _require_msr_reader():
    if OBFFile is None:
        raise AttributeError(_MSR_READER_INSTALL_ERROR)


def _read_msr_stack(msr_path: PathLike, stack_index: int = 0) -> np.ndarray:
    _require_msr_reader()
    msr_path = os.fspath(msr_path)
    with OBFFile(msr_path) as msr:
        if stack_index < 0 or stack_index >= msr.num_stacks:
            raise IndexError(
                f"Invalid stack index {stack_index} for {msr_path}; available stacks: 0..{msr.num_stacks - 1}"
            )
        image = msr.read_stack(stack_index)
    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D MSR stack from {msr_path}, got shape {image.shape}"
        )
    return image


def _normalize_stack_selection(
    stack_selection: StackSelection,
) -> tuple[Union[int, str], ...]:
    if isinstance(stack_selection, (str, int)):
        return (stack_selection,)
    stack_selection = tuple(stack_selection)
    if not stack_selection:
        raise ValueError("At least one MSR stack index is required")
    return stack_selection


def _normalize_stack_indices(stack_index: StackIndexSelection = 0) -> tuple[int, ...]:
    if isinstance(stack_index, int):
        return (stack_index,)
    stack_index = tuple(stack_index)
    if not stack_index:
        raise ValueError("At least one MSR stack index is required")
    if any(not isinstance(index, int) for index in stack_index):
        raise TypeError("stack_index must contain integer stack ids only")
    return stack_index


def _normalize_stack_names(stack_names: StackNameSelection) -> tuple[str, ...]:
    if isinstance(stack_names, str):
        return (stack_names,)
    stack_names = tuple(stack_names)
    if not stack_names:
        raise ValueError("At least one MSR stack name is required")
    if any(not isinstance(name, str) for name in stack_names):
        raise TypeError("stack_names must contain string stack names only")
    return stack_names


def _resolve_stack_indices(msr, stack_selection: StackSelection) -> tuple[int, ...]:
    resolved = []
    for stack in _normalize_stack_selection(stack_selection):
        if isinstance(stack, int):
            resolved.append(stack)
        else:
            # msr.stack_names should be list[str]
            if stack not in msr.stack_names:
                raise KeyError(
                    f"Stack name '{stack}' not found in MSR file; available stacks: {msr.stack_names}"
                )
            resolved.append(msr.stack_names.index(stack))
    return tuple(resolved)


def _get_msr_stack_shape(
    msr_path: PathLike, stack_selection: StackSelection = 0
) -> tuple[int, int]:
    _require_msr_reader()
    with OBFFile(os.fspath(msr_path)) as msr:
        stack_index = _resolve_stack_indices(msr, stack_selection)[0]
    return tuple(_read_msr_stack(msr_path, stack_index=stack_index).shape)


def _resolve_stack_indices_for_path(
    msr_path: PathLike, stack_selection: StackSelection
) -> tuple[int, ...]:
    _require_msr_reader()
    with OBFFile(os.fspath(msr_path)) as msr:
        return _resolve_stack_indices(msr, stack_selection)


class MSRSampleCollection:
    """Collection-like helper for loading one sample per MSR file."""

    def __init__(
        self,
        image_paths: Sequence[PathLike],
        stack_index: StackIndexSelection = 0,
        stack_names: StackNameSelection | None = None,
    ):
        self.image_paths = [os.fspath(path) for path in image_paths]
        if stack_names is not None and stack_index != 0:
            raise ValueError("Pass either stack_index or stack_names, not both.")
        if stack_names is not None:
            self.stack_selection = _normalize_stack_names(stack_names)
        else:
            self.stack_selection = _normalize_stack_indices(stack_index)
        self.stack_indices = _resolve_stack_indices_for_path(
            self.image_paths[0], self.stack_selection
        )
        sample = _read_msr_stack(self.image_paths[0], self.stack_indices[0])
        self._dtype = sample.dtype
        self._shape = (
            sample.shape
            if len(self.stack_indices) == 1
            else (len(self.stack_indices),) + tuple(sample.shape)
        )

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def chunks(self):
        return None

    @property
    def size(self):
        return int(np.prod(self._shape))

    @property
    def attrs(self):
        return {}

    def read_sample(self, index: int) -> np.ndarray:
        stack_indices = _resolve_stack_indices_for_path(
            self.image_paths[index], self.stack_selection
        )
        if len(stack_indices) == 1:
            return _read_msr_stack(self.image_paths[index], stack_indices[0])
        data = [
            _read_msr_stack(self.image_paths[index], stack_index)
            for stack_index in stack_indices
        ]
        return np.stack(data, axis=0)


class MSRSampleCollectionPerSampleSelection:
    """
    Changed SampleCollection that allows to specify stack selection per sample, which is required for
    dynamic stack selection.

    For some datasets it does not make sense to define stack_index, stack_names globally for all datasets.
    Because of human error stack order is not the same across samples, so stack index 0 in one sample may correspond to
    a different stack than stack index 0 in another sample.
    Stack names in most systems are not defined automatically and people need to type them manually, which
    can lead to human error. Same for the stack names, sometimes basic spelling can cause the mismatch between
    global stack_names, e.g. 'Mic60_STED' or 'MIc60_STeD'.
    Sometimes stack_name also include fluorophore name like 'dsDNA_StarRed_STED {0}', sometimes not.

    Additional bug is when a measurement is started in Abberior software, the measurement file will save all open
    windows, among others also plots that will have dimensions [N, 1, 1] and usually have "Pop" in the stack name.
    These stacks also must be excluded.
    """

    def __init__(
        self,
        image_paths: Sequence[PathLike],
        default_stack_selection: StackIndexSelection = 0,
    ):
        self.image_paths = [os.fspath(path) for path in image_paths]
        # next lines are only done to get dtype of the first sample,
        # alternately, dtype can be given as an argument to the constructor and these 4 lines deleted.
        stack_selection = _normalize_stack_indices(default_stack_selection)
        stack_indices = _resolve_stack_indices_for_path(
            self.image_paths[0], stack_selection
        )
        sample = _read_msr_stack(self.image_paths[0], stack_indices[0])
        self._dtype = sample.dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        # ndim depend from shape, which is sample-dependent due to dynamic stack selection.
        return None

    @property
    def shape(self):
        # Shape is sample-dependent due to dynamic stack selection.
        return None

    @property
    def chunks(self):
        return None

    @property
    def size(self):
        # size depend from shape, which is sample-dependent due to dynamic stack selection.
        return None

    @property
    def attrs(self):
        return {}

    def read_sample(self, index: int, stack_selection: StackSelection) -> np.ndarray:
        """
        This function must be called from getitem of MSRDataset, with specified stack_selection.
        Options:
            for stack_index selection: tuple of ints of indexes of msr stacks to load.
            for stack_name selection: tuple of strings of names of msr stacks to load.
        """
        img_path = self.image_paths[index]
        # next function will anyway check if the stack names or indexes are in correct format, dont need to do it here.
        stack_indices = _resolve_stack_indices_for_path(img_path, stack_selection)
        data = [_read_msr_stack(img_path, stack_index) for stack_index in stack_indices]
        if len(data) == 1:
            return data[0]
        return np.stack(data, axis=0)


class MSRFile(Mapping):
    """Root object for a file handle representing an MSR file."""

    def __init__(self, path: PathLike, mode: str = "r"):
        _require_msr_reader()
        if mode != "r":
            raise ValueError("MSR files only support read mode.")
        self.path = os.fspath(path)
        self.mode = mode
        self.msr = OBFFile(self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.msr.close()

    @property
    def attrs(self):
        return {}

    def _normalize_key(self, key):
        if key == "data":
            return key
        try:
            return int(key)
        except (TypeError, ValueError):
            if key in self.msr.stack_names:
                return self.msr.stack_names.index(key)
            raise KeyError(f"Could not find key {key}")

    def __getitem__(self, key):
        key = self._normalize_key(key)
        if key == "data":
            return MSRDataset(self.path, tuple(range(self.msr.num_stacks)))
        return MSRDataset(self.path, key)

    def __iter__(self):
        for index in range(self.msr.num_stacks):
            yield str(index)
        yield "data"

    def __len__(self):
        return self.msr.num_stacks + 1

    def __contains__(self, name):
        if name == "data":
            return True
        try:
            index = int(name)
            return 0 <= index < self.msr.num_stacks
        except (TypeError, ValueError):
            return name in self.msr.stack_names


class MSRDataset:
    """Dataset object for one or more stacks in a single MSR file."""

    def __init__(self, path: PathLike, stack_selection: StackSelection):
        self.path = os.fspath(path)
        self.stack_selection = _normalize_stack_selection(stack_selection)

        with OBFFile(self.path) as msr:
            self.stack_indices = _resolve_stack_indices(msr, self.stack_selection)
            sample = msr.read_stack(self.stack_indices[0])

        if sample.ndim != 2:
            raise ValueError(
                f"Expected a 2D MSR stack from {self.path}, got shape {sample.shape}"
            )

        self._dtype = sample.dtype
        self._shape = (
            sample.shape
            if len(self.stack_indices) == 1
            else (len(self.stack_indices),) + sample.shape
        )
        self._size = int(np.prod(self._shape))

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def chunks(self):
        return None

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def attrs(self):
        return {}

    def _read_stack(self, stack_index: int, spatial_index):
        with OBFFile(self.path) as msr:
            data = msr.read_stack(stack_index)
        return data[spatial_index]

    def __getitem__(self, key):
        key, to_squeeze = normalize_index(key, self.shape)
        if len(self.stack_indices) == 1:
            data = self._read_stack(self.stack_indices[0], key)
            return squeeze_singletons(data, to_squeeze).copy()

        channel_index, spatial_index = key[0], key[1:]
        channel_step = 1 if channel_index.step is None else channel_index.step
        data = [
            self._read_stack(stack_index, spatial_index)
            for stack_index in self.stack_indices[
                channel_index.start : channel_index.stop : channel_step
            ]
        ]
        return squeeze_singletons(np.stack(data, axis=0), to_squeeze).copy()
