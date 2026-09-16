"""Structured experiment logging protocol."""

from typing import Any, Dict, Optional, Protocol, Union


class ExperimentLogger(Protocol):
    """Protocol for experiment loggers."""

    def start_run(self, run_name: str, config: dict[str, Any]) -> None:
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        ...

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        ...

    def log_histogram(self, name: str, values: Any, step: Optional[int] = None) -> None:
        ...

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        ...

    def end_run(self) -> None:
        ...


class NullExperimentLogger:
    """No-op logger for when logging is disabled."""

    def start_run(self, run_name: str, config: dict[str, Any]) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        pass

    def log_histogram(self, name: str, values: Any, step: Optional[int] = None) -> None:
        pass

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        pass

    def end_run(self) -> None:
        pass
