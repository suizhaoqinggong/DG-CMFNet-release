"""Generic component registry."""

from typing import Callable, Generic, TypeVar

T = TypeVar("T")
Factory = Callable[..., T]


class RegistryError(Exception):
    """Base exception for registry errors."""


class DuplicateRegistrationError(RegistryError):
    """Raised when attempting to register an existing name."""


class UnknownRegistrationError(RegistryError):
    """Raised when looking up an unregistered name."""


class Registry(Generic[T]):
    """A type-safe component registry."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, Factory[T]] = {}

    def _normalize(self, name: str) -> str:
        return name.lower().strip()

    def register(self, name: str, factory: Factory[T]) -> None:
        """Register a factory under the given name."""
        key = self._normalize(name)
        if key in self._entries:
            raise DuplicateRegistrationError(
                f"{self._kind} '{key}' is already registered. "
                f"Existing: {self._entries[key]}, New: {factory}"
            )
        self._entries[key] = factory

    def decorator(self, name: str) -> Callable[[Factory[T]], Factory[T]]:
        """Return a decorator that registers the decorated factory."""
        def _decorator(factory: Factory[T]) -> Factory[T]:
            self.register(name, factory)
            return factory

        return _decorator

    def get(self, name: str) -> Factory[T]:
        """Get a factory by name."""
        key = self._normalize(name)
        if key not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise UnknownRegistrationError(
                f"Unknown {self._kind} '{key}'. Available: {available}"
            )
        return self._entries[key]

    def create(self, name: str, **kwargs: object) -> T:
        """Instantiate a component by name with kwargs."""
        factory = self.get(name)
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        """Return all registered names."""
        return tuple(sorted(self._entries.keys()))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self._normalize(name) in self._entries
