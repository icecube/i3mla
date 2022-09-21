"""Docstring"""

from typing import ClassVar, List, Tuple, Type, TypeVar
import dataclasses

import numpy as np
import numpy.typing as npt


T = TypeVar('T', bound='Events')
@dataclasses.dataclass(kw_only=True)
class Events():
    """Docstring"""

    # dtype_name: (dtype_val, idx, required)
    _dtype_map: ClassVar[dict] = {
        'run': (np.dtype('uint64'), 0, True),
        'event': (np.dtype('uint64'), 1, True),
        'subevent': (np.dtype('uint64'), 2, True),
        'ra': (np.dtype('float64'), 0, True),
        'dec': (np.dtype('float64'), 1, True),
        'azi': (np.dtype('float64'), 2, True),
        'zen': (np.dtype('float64'), 3, True),
        'time': (np.dtype('float64'), 4, True),
        'logE': (np.dtype('float64'), 5, True),
        'angErr': (np.dtype('float64'), 6, True),
        'sinDec': (np.dtype('float64'), 7, False),
    }

    _ints: npt.NDArray[np.uint64]
    _floats: npt.NDArray[np.float64]

    @classmethod
    def _get_n_dtypes(cls) -> Tuple[int, int]:
        n_ints = sum([
            val == np.dtype('uint64') for _, (val, _, _) in cls._dtype_map.items()])
        n_floats = sum([
            val == np.dtype('float64') for _, (val, _, _) in cls._dtype_map.items()])
        return (n_ints, n_floats)

    @classmethod
    def empty(cls: Type[T]) -> T:
        """Docstring"""
        n_ints, n_floats = cls._get_n_dtypes()
        return cls(
            _ints=np.empty((n_ints, 0), dtype=np.uint64),
            _floats=np.empty((n_floats, 0), dtype=np.float64),
        )


    @classmethod
    def from_nusources_events(cls: Type[T], nu_sources_events: np.ndarray) -> T:
        """Docstring"""
        n_ints, n_floats = cls._get_n_dtypes()
        ints = np.empty((n_ints, len(nu_sources_events)), dtype=np.uint64)
        floats = np.empty((n_floats, len(nu_sources_events)), dtype=np.float64)
        optionals_to_generate = []

        for dtype_name, (dtype_val, idx, req) in cls._dtype_map.items():
            if dtype_name not in nu_sources_events.dtype.fields:
                if req:
                    raise KeyError(
                        f'Missing dtype {dtype_name} in input structured array.')
                else:
                    optionals_to_generate.append(dtype_name)
                    continue
            if not dtype_val == nu_sources_events.dtype.fields[dtype_name][0]:
                if nu_sources_events.dtype.fields[dtype_name][0] == np.dtype('uint8'):
                    ints[idx, :] = nu_sources_events[dtype_name].astype(np.uint64)
                elif nu_sources_events.dtype.fields[dtype_name][0] == np.dtype('uint32'):
                    ints[idx, :] = nu_sources_events[dtype_name].astype(np.uint64)
                elif nu_sources_events.dtype.fields[dtype_name][0] == np.dtype('float32'):
                    floats[idx, :] = nu_sources_events[dtype_name].astype(np.float64)
                else:    
                    raise ValueError(''.join([
                        f'Dtype of {dtype_name} in input structured array is ',
                        f'{nu_sources_events.dtype.fields[dtype_name][0]} when it should ',
                        f'be {dtype_val}.',
                    ]))
            elif dtype_val == np.dtype('uint64'):
                ints[idx, :] = nu_sources_events[dtype_name]
            elif dtype_val == np.dtype('float64'):
                floats[idx, :] = nu_sources_events[dtype_name]

        for dtype_name in optionals_to_generate:
            ints, floats = cls._generate_optional_field(ints, floats, dtype_name)

        return cls(_ints=ints, _floats=floats)

    def _get_int(self, dtype_name: str) -> npt.NDArray[np.uint64]:
        return np.squeeze(self._ints[self.__class__._dtype_map[dtype_name][1], :])

    def _get_float(self, dtype_name: str) -> npt.NDArray[np.float64]:
        return np.squeeze(self._floats[self.__class__._dtype_map[dtype_name][1], :])

    def _set_float(self, dtype_name: str, arr: npt.NDArray[np.float64]) -> None:
        self._floats[self.__class__._dtype_map[dtype_name][1], :] = arr

    @property
    def run(self) -> npt.NDArray[np.uint64]:
        return self._get_int('run')

    @property
    def event(self) -> npt.NDArray[np.uint64]:
        return self._get_int('event')

    @property
    def subevent(self) -> npt.NDArray[np.uint64]:
        return self._get_int('subevent')

    @property
    def ra(self) -> npt.NDArray[np.float64]:
        return self._get_float('ra')

    @property
    def dec(self) -> npt.NDArray[np.float64]:
        return self._get_float('dec')

    @property
    def azi(self) -> npt.NDArray[np.float64]:
        return self._get_float('azi')

    @property
    def zen(self) -> npt.NDArray[np.float64]:
        return self._get_float('zen')

    @property
    def time(self) -> npt.NDArray[np.float64]:
        return self._get_float('time')

    @property
    def logE(self) -> npt.NDArray[np.float64]:
        return self._get_float('logE')

    @property
    def angErr(self) -> npt.NDArray[np.float64]:
        return self._get_float('angErr')

    @property
    def sinDec(self) -> npt.NDArray[np.float64]:
        return self._get_float('sinDec')

    @time.setter
    def time(self, arr: npt.NDArray[np.float64]) -> None:
        self._set_float('time', arr)

    @ra.setter
    def ra(self, arr: npt.NDArray[np.float64]) -> None:
        self._set_float('ra', arr)

    @dec.setter
    def dec(self, arr: npt.NDArray[np.float64]) -> None:
        self._set_float('dec', arr)
        self._set_float('sinDec', np.sin(arr))

    @classmethod
    def _generate_optional_field(
        cls,
        ints: np.ndarray,
        floats: np.ndarray,
        dtype_name: str,
    ) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.float64]]:
        """Docstring"""
        if dtype_name == 'sinDec':
            floats[cls._dtype_map['sinDec'][1], :] = np.sin(
                floats[cls._dtype_map['dec'][1], :])
        return ints, floats

    def sample(self: T, n: int, rng: np.random.Generator) -> T:
        """Docstring"""
        return self.from_idx(rng.choice(self._ints.shape[-1], n))

    def from_idx(self: T, idx: np.ndarray) -> T:
        return self.__class__(
            _ints=np.take(self._ints, idx, axis=-1),
            _floats=np.take(self._floats, idx, axis=-1),
        )

    def copy(self: T) -> T:
        return self.__class__(
            _ints=self._ints.copy(),
            _floats=self._floats.copy(),
        )

    def sort(self, key) -> None:
        if self.__class__._dtype_map[key][0] == np.dtype('uint64'):
            idxs = np.argsort(self._ints[self.__class__._dtype_map[key][1], :])
        else:
            idxs = np.argsort(self._floats[self.__class__._dtype_map[key][1], :])
        self._ints = self._ints[:, idxs]
        self._floats = self._floats[:, idxs]

    def __len__(self):
        return self._ints.shape[-1]

    @classmethod
    def concatenate(cls, events_list: List['Events']) -> 'Events':
        return cls(
            _ints=np.concatenate([events._ints for events in events_list], axis=-1),
            _floats=np.concatenate([events._floats for events in events_list], axis=-1),
        )


@dataclasses.dataclass(kw_only=True)
class DataEvents(Events):
    """Docstring"""

S = TypeVar('S', bound='SimEvents')
@dataclasses.dataclass(kw_only=True)
class SimEvents(Events):
    """Docstring"""
    _dtype_map: ClassVar[dict] = {
        **Events._dtype_map,
        'trueRa': (np.dtype('float64'), 8, True),
        'trueDec': (np.dtype('float64'), 9, True),
        'trueE': (np.dtype('float64'), 10, True),
        'ow': (np.dtype('float64'), 11, True),
        'weight': (np.dtype('float64'), 12, False),
    }

    @classmethod
    def _generate_optional_field(
        cls,
        ints: np.ndarray,
        floats: np.ndarray,
        dtype_name: str,
    ) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.float64]]:
        if dtype_name == 'weight':
            floats[cls._dtype_map[dtype_name][1], :] = np.ones(ints.shape[-1])
        return super(SimEvents, cls)._generate_optional_field(ints, floats, dtype_name)

    def sample(self: S, n: int, rng: np.random.Generator) -> S:
        p = self._floats[self.__class__._dtype_map['weight'][1], :] / self._floats[
            self.__class__._dtype_map['weight'][1], :].sum()
        return self.from_idx(rng.choice(self._ints.shape[-1], n, replace=False, p=p))

    @property
    def trueRa(self) -> npt.NDArray[np.float64]:
        return self._get_float('trueRa')

    @property
    def trueDec(self) -> npt.NDArray[np.float64]:
        return self._get_float('trueDec')

    @property
    def trueE(self) -> npt.NDArray[np.float64]:
        return self._get_float('trueE')

    @property
    def ow(self) -> npt.NDArray[np.float64]:
        return self._get_float('ow')

    @property
    def weight(self) -> npt.NDArray[np.float64]:
        return self._get_float('weight')

    @weight.setter
    def weight(self, arr: npt.NDArray[np.float64]) -> None:
        self._set_float('weight', arr)

    @ow.setter
    def ow(self, arr: npt.NDArray[np.float64]) -> None:
        self._set_float('ow', arr)

    @trueRa.setter
    def trueRa(self, arr: npt.NDArray[np.float64]) -> None:
        self._set_float('trueRa', arr)

    @trueDec.setter
    def trueDec(self, arr: npt.NDArray[np.float64]) -> None:
        self._set_float('trueDec', arr)

    def to_events(self) -> Events:
        # rewrite in a more maintainable, but still fast way
        return Events(_ints=self._ints, _floats=self._floats[:8, :])

