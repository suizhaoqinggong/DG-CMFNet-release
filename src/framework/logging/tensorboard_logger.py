"""TensorBoard experiment logger."""

from typing import Any, Dict, Optional, Union

from torch.utils.tensorboard import SummaryWriter

from framework.logging.structured_logger import ExperimentLogger


class TensorBoardLogger:
    """Logger that writes to TensorBoard."""

    def __init__(self, log_dir: str) -> None:
        self.writer = SummaryWriter(log_dir)

    def start_run(self, run_name: str, config: dict[str, Any]) -> None:
        self.writer.add_text("run_name", run_name)
        for key, value in config.items():
            self.writer.add_text(f"config/{key}", str(value))

    def log_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            self.writer.add_text(f"param/{key}", str(value))

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step or 0)

    def log_histogram(self, name: str, values: Any, step: Optional[int] = None) -> None:
        self.writer.add_histogram(name, values, step or 0)

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        self.writer.add_figure(name, figure, step or 0)

    def end_run(self) -> None:
        self.writer.close()
