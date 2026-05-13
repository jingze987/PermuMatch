"""Inference-only model builder for MeViS.

The public entry point is `build_model(args)`, which returns a GroundingDINO-based segmentation model ready for checkpoint loading and MeViS inference. Supervision-time builders and optimization utilities are not included.
"""

import torch

from .GroundingDINO import build_groundingdino


def build_model(args):
    """Build the MeViS inference model only.

    Args:
        args: EasyDict loaded from `configs/mevis_swinb.yaml` and CLI args.

    Returns:
        torch.nn.Module: model with the `infer(...)` method used by
        `eval/inference_mevis.py`.
    """
    dataset_name = getattr(args, "dataset_name", "mevis")
    if dataset_name != "mevis":
        raise ValueError(
            f"This cleaned package only supports MeViS inference; got dataset_name={dataset_name!r}."
        )

    num_classes = 1
    args.GroundingDINO.single_frame = False

    model = build_groundingdino(args.GroundingDINO, num_classes=num_classes)
    model.to(torch.device(args.device))
    model.eval()
    return model
