"""CLI entry point for the training framework."""

from __future__ import annotations

import argparse
from pathlib import Path

from framework.core.checkpoint import CheckpointManager
from framework.core.distributed import DistributedContext, broadcast_object, cleanup_distributed, setup_distributed
from framework.core.run_layout import (
    RunLayout,
    layout_from_run_dir,
    resolve_run_layout,
    snapshot_config,
    snapshot_split_ids,
)
from framework.core.seed import set_global_seed
from framework.core.trainer import Trainer, TrainerSettings
from framework.logging.structured_logger import NullExperimentLogger
from framework.logging.tensorboard_logger import TensorBoardLogger
from framework.registry.defaults import RegistryBundle
from framework.registry.factories import (
    build_component_bundle,
    create_default_registries,
    load_merged_experiment_config,
    optional_int,
    require_list,
    require_string,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="framework",
        description="Deep Learning Training Framework",
    )
    parser.add_argument(
        "command",
        choices=["train", "validate", "test", "fit"],
        help="Command to run",
    )
    parser.add_argument(
        "--configs",
        required=True,
        help="Path to experiment config TOML",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model config TOML",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint (for validate/test)",
    )
    return parser.parse_args()


def build_trainer(
    config: dict,
    registries: RegistryBundle,
    distributed_context: DistributedContext | None = None,
) -> tuple[Trainer, RunLayout]:
    """Build trainer and run layout from config."""
    exp_section = config.get("experiment", {})
    seed = optional_int(exp_section.get("seed"))
    if seed is not None:
        set_global_seed(seed)

    # Build component bundle
    bundle = build_component_bundle(config, registries, distributed_context=distributed_context)

    # Experiment metadata
    experiment_name = require_string(exp_section, "name", "[experiment]")
    model_section = config.get("model", {})
    model_name = require_string(model_section, "name", "[model]")
    class_names = require_list(exp_section, "class_names", "[experiment]")

    # Run layout
    root_dir = config.get("output", {}).get("root_dir", "runs")
    if distributed_context is not None and distributed_context.enabled:
        run_dir = None
        if distributed_context.is_main_process:
            run_dir = str(
                resolve_run_layout(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    root_dir=root_dir,
                ).run_dir
            )
        run_dir = broadcast_object(run_dir, distributed_context)
        layout = layout_from_run_dir(run_dir)
    else:
        layout = resolve_run_layout(
            experiment_name=experiment_name,
            model_name=model_name,
            root_dir=root_dir,
        )

    # Snapshot config
    if distributed_context is None or distributed_context.is_main_process:
        snapshot_config(config, layout.run_dir)
        snapshot_split_ids(bundle.data_adapter, layout.run_dir)

    # Trainer settings
    trainer_section = config.get("trainer", {})
    settings = TrainerSettings(
        epochs=int(trainer_section.get("epochs", 100)),
        patience=int(trainer_section.get("patience", 10)),
        device=trainer_section.get("device", "auto"),
        amp=bool(trainer_section.get("amp", False)),
        grad_clip_norm=float(v) if (v := trainer_section.get("grad_clip_norm")) is not None else None,
        seed=seed,
        lr_scheduler=bool(trainer_section.get("lr_scheduler", False)),
        lr_scheduler_type=trainer_section.get("lr_scheduler_type", "reduce_on_plateau"),
        lr_scheduler_factor=float(trainer_section.get("lr_scheduler_factor", 0.5)),
        lr_scheduler_patience=int(trainer_section.get("lr_scheduler_patience", 10)),
        lr_scheduler_power=float(trainer_section.get("lr_scheduler_power", 0.9)),
    )

    # Checkpoint manager
    checkpoint_section = config.get("checkpoint", {})
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=layout.checkpoint_dir,
        monitor=checkpoint_section.get("monitor", "val_loss"),
        mode=checkpoint_section.get("mode", "min"),
    )

    # Logger
    if distributed_context is None or distributed_context.is_main_process:
        logger = TensorBoardLogger(str(layout.log_dir))
        logger.start_run(
            run_name=f"{experiment_name}_{model_name}",
            config=config,
        )
    else:
        logger = NullExperimentLogger()

    trainer = Trainer(
        model=bundle.model,
        task=bundle.task,
        optimizer=bundle.optimizer,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        test_loader=bundle.test_loader,
        train_metrics=bundle.train_metrics,
        val_metrics=bundle.val_metrics,
        test_metrics=bundle.test_metrics,
        settings=settings,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        run_dir=layout.run_dir,
        distributed_context=distributed_context,
    )

    return trainer, layout


def main() -> int:
    args = parse_args()

    # Load merged config
    config = load_merged_experiment_config(args.configs, args.model)
    distributed_context = setup_distributed(config.get("trainer", {}).get("device", "auto"))

    trainer = None
    layout = None
    try:
        # Create registries
        registries = create_default_registries()

        # Build trainer
        trainer, layout = build_trainer(config, registries, distributed_context=distributed_context)

        # Execute command
        if args.command == "train":
            trainer.train()
        elif args.command == "validate":
            trainer.validate(checkpoint_path=args.checkpoint)
        elif args.command == "test":
            trainer.test(checkpoint_path=args.checkpoint)
        elif args.command == "fit":
            trainer.fit()
        else:
            raise ValueError(f"Unknown command: {args.command}")

        if distributed_context.is_main_process and layout is not None:
            print(f"\nRun outputs saved to: {layout.run_dir}")
        return 0
    finally:
        if trainer is not None:
            trainer.logger.end_run()
        cleanup_distributed(distributed_context)


if __name__ == "__main__":
    raise SystemExit(main())
