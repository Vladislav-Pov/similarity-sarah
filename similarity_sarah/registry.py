"""A tiny name -> class registry for swappable components.

The extensibility goal of the rewrite is "a new baseline = two files"
(``algorithms/<name>.py`` + ``configs/algorithm/<name>.yaml``) with no edits to
the runner. A class self-registers with a decorator::

    @ALGORITHMS.register("my_algo")
    class MyAlgo(Algorithm):
        ...

and the runner resolves it by name via :meth:`Registry.get` / :meth:`create`,
replacing the hard-coded ``if name == ...`` dispatch ladders.

Registry singletons live next to the base classes they hold (e.g. the prox and
algorithm packages), not here, so this module stays dependency-free.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Case-insensitive registry mapping names to classes of one kind."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, *names: str) -> Callable[[type[T]], type[T]]:
        """Decorator registering a class under one or more ``names``.

        Raises
        ------
        ValueError
            If a name is empty or already registered.
        """

        def decorate(cls: type[T]) -> type[T]:
            for name in names:
                key = name.lower()
                if not key:
                    raise ValueError(f"empty {self._kind} name")
                if key in self._entries:
                    raise ValueError(f"{self._kind} {name!r} already registered")
                self._entries[key] = cls
            return cls

        return decorate

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name``.

        Raises
        ------
        KeyError
            If ``name`` is not registered (the message lists known names).
        """
        key = name.lower()
        try:
            return self._entries[key]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "<none>"
            raise KeyError(
                f"unknown {self._kind} {name!r}; registered: {known}"
            ) from None

    def create(self, name: str, /, *args: object, **kwargs: object) -> T:
        """Instantiate the class registered under ``name``."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        """Sorted list of registered names."""
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._entries
