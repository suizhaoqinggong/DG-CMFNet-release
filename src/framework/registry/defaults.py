"""Default registry bundle with built-in components."""

from dataclasses import dataclass

from framework.contracts.data import DataAdapter
from framework.contracts.metric import Metric
from framework.contracts.model import ModelAdapter
from framework.contracts.task import Task

from .registry import Registry


def register_builtin_data_adapters(registry: Registry[DataAdapter]) -> None:
    """Register built-in data adapters."""
    from data_adapters.brats_adapter import BraTS2DDataAdapter, BraTSDataAdapter
    from data_adapters.dummy_adapter import DummyDataAdapter

    registry.register("dummy", DummyDataAdapter)
    registry.register("brats", BraTSDataAdapter)
    registry.register("brats_2d", BraTS2DDataAdapter)


def register_builtin_tasks(registry: Registry[Task]) -> None:
    """Register built-in tasks."""
    from tasks.classification import ClassificationTask
    from tasks.segmentation import SegmentationTask

    registry.register("classification", ClassificationTask)
    registry.register("segmentation", SegmentationTask)


def register_builtin_metrics(registry: Registry[Metric]) -> None:
    """Register built-in metrics."""
    from framework.metrics.accuracy import AccuracyMetric
    from framework.metrics.auc import AUCMetric
    from framework.metrics.brats import BraTSRegionDiceMetric, BraTSRegionHausdorffMetric
    from framework.metrics.dice import DiceMetric
    from framework.metrics.iou import IoUMetric
    from framework.metrics.precision_recall_f1 import PrecisionRecallF1Metric

    registry.register("accuracy", AccuracyMetric)
    registry.register("auc", AUCMetric)
    registry.register("precision_recall_f1", PrecisionRecallF1Metric)
    registry.register("dice", DiceMetric)
    registry.register("iou", IoUMetric)
    registry.register("brats_dice", BraTSRegionDiceMetric)
    registry.register("brats_hd", BraTSRegionHausdorffMetric)


@dataclass(frozen=True)
class RegistryBundle:
    """Holds all component registries."""

    data_adapters: Registry[DataAdapter]
    models: Registry[ModelAdapter]
    tasks: Registry[Task]
    metrics: Registry[Metric]


def create_default_registries() -> RegistryBundle:
    """Create and populate default registries."""
    from models.catalog import register_models

    data_registry: Registry[DataAdapter] = Registry("data_adapter")
    model_registry: Registry[ModelAdapter] = Registry("model")
    task_registry: Registry[Task] = Registry("task")
    metric_registry: Registry[Metric] = Registry("metric")

    register_builtin_data_adapters(data_registry)
    register_models(model_registry)
    register_builtin_tasks(task_registry)
    register_builtin_metrics(metric_registry)

    return RegistryBundle(
        data_adapters=data_registry,
        models=model_registry,
        tasks=task_registry,
        metrics=metric_registry,
    )
