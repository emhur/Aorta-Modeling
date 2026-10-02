"""
Segment the aorta with nnInteractive using point prompts sampled from the ground-truth mask.

For one AVT case (e.g. AVT_dataset/KiTS/K1/K1.nrrd + K1.seg.nrrd) this script:
  1. loads the CT volume and its ground-truth aorta mask,
  2. samples N positive points spread evenly along the aorta (each placed deep inside the
     vessel, at the maximum of the in-slice distance transform),
  3. optionally samples negative points from a thin ring just outside the aorta,
  4. feeds the points to nnInteractive one by one,
  5. saves the predicted mask (with the CT's geometry) and reports Dice against the GT.

Usage:
  conda activate comsail
  python segment_aorta_points.py --case AVT_dataset/KiTS/K1 --n-pos 10 --n-neg 0
"""
import argparse
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # run the rare ops MPS lacks on CPU instead of erroring
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

# Libraries to visualize 3D mesh result as we build it
import pyvista as pv
import matplotlib

import nnInteractive.inference.inference_session as nni_session
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.model_management import ensure_model_available, get_default_model_id


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def enable_mps_fp16():
    """nnInteractive only uses mixed precision on CUDA and runs fp32 elsewhere. On Apple GPUs
    fp16 autocast makes the forward pass ~5x faster with identical argmax output, so swap the
    no-op context it uses for non-CUDA devices with an MPS fp16 autocast."""
    nni_session.dummy_context = lambda: torch.autocast("mps", dtype=torch.float16)


def find_case_files(case_dir: Path):
    seg = next(case_dir.glob("*.seg.nrrd"))
    img = next(p for p in case_dir.glob("*.nrrd") if not p.name.endswith(".seg.nrrd"))
    return img, seg


def sample_positive_points(gt: np.ndarray, n_points: int):
    """Pick n_points inside the mask, spread evenly along the first array axis (z).

    On each chosen slice, the point is placed at the maximum of the distance transform of
    the largest connected component, i.e. as far from the vessel wall as possible.
    """
    z_with_mask = np.where(gt.any(axis=(1, 2)))[0]
    # Evenly spaced slices, skipping the very ends of the mask where the vessel is tiny.
    margin = max(1, len(z_with_mask) // (4 * n_points))
    idx = np.linspace(margin, len(z_with_mask) - 1 - margin, n_points).round().astype(int)
    points = []
    for z in z_with_mask[np.unique(idx)]:
        labels, n = ndimage.label(gt[z])
        if n > 1:
            largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
            slice_mask = labels == largest
        else:
            slice_mask = labels > 0
        dist = ndimage.distance_transform_edt(slice_mask)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        points.append((int(z), int(y), int(x)))
    return points


def sample_negative_points(gt: np.ndarray, n_points: int, ring_voxels: int, rng: np.random.Generator):
    """Pick n_points at random from a thin shell just outside the mask."""
    if n_points == 0:
        return []
    struct = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(gt, struct, iterations=ring_voxels)
    ring = dilated & ~ndimage.binary_dilation(gt, struct, iterations=max(1, ring_voxels // 2))
    candidates = np.argwhere(ring)
    chosen = candidates[rng.choice(len(candidates), size=min(n_points, len(candidates)), replace=False)]
    return [tuple(int(c) for c in p) for p in chosen]


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / denom if denom else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", type=Path, default=Path("AVT_dataset/KiTS/K1"),
                        help="Case folder containing <name>.nrrd and <name>.seg.nrrd")
    parser.add_argument("--n-pos", type=int, default=10, help="Number of positive points from the GT mask")
    parser.add_argument("--n-neg", type=int, default=0, help="Number of negative points just outside the GT mask")
    parser.add_argument("--neg-ring", type=int, default=6, help="Distance (voxels) of the negative-point shell")
    parser.add_argument("--device", default="auto", help="auto | cuda:0 | mps | cpu")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 autocast on MPS (slower)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="Output path (default: <case>/<name>.pred.nrrd)")
    args = parser.parse_args()

    img_path, seg_path = find_case_files(args.case)
    out_path = args.out or args.case / img_path.name.replace(".nrrd", ".pred.nrrd")
    rng = np.random.default_rng(args.seed)

    # --- Load data (no intensity preprocessing: nnInteractive wants the raw image) ---
    print(f"Loading {img_path}")
    image_sitk = sitk.ReadImage(str(img_path))
    img = sitk.GetArrayFromImage(image_sitk)[None]  # (1, z, y, x)

    seg_sitk = sitk.ReadImage(str(seg_path))
    gt = sitk.GetArrayFromImage(seg_sitk) > 0  # aorta is label 1
    if gt.shape != img.shape[1:]:
        # Slicer .seg.nrrd files can be cropped to the segment extent; resample onto the CT grid.
        seg_sitk = sitk.Resample(seg_sitk, image_sitk, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        gt = sitk.GetArrayFromImage(seg_sitk) > 0
    print(f"Image shape {img.shape[1:]}, GT aorta voxels: {gt.sum()}")

    # --- Sample prompts from the ground truth ---
    pos_points = sample_positive_points(gt, args.n_pos)
    neg_points = sample_negative_points(gt, args.n_neg, args.neg_ring, rng)
    print(f"Positive points (z, y, x): {pos_points}")
    if neg_points:
        print(f"Negative points (z, y, x): {neg_points}")

    # --- Set up nnInteractive ---
    device = pick_device(args.device)
    if device.type == "mps" and not args.no_fp16:
        enable_mps_fp16()
    print(f"Using device: {device}{' (fp16 autocast)' if device.type == 'mps' and not args.no_fp16 else ''}")
    model_path = ensure_model_available(get_default_model_id())
    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
    )
    session.initialize_from_trained_model_folder(str(model_path))
    session.set_image(img)
    target = torch.zeros(img.shape[1:], dtype=torch.uint8)
    session.set_target_buffer(target)

    # --- Interact: one prediction per point, so AutoZoom refines around each prompt ---
    prompts = [(p, True) for p in pos_points] + [(p, False) for p in neg_points]
    for i, (point, positive) in enumerate(prompts, 1):
        t0 = time.time()
        session.add_point_interaction(point, include_interaction=positive)
        pred = target.numpy()
        print(f"[{i}/{len(prompts)}] {'+' if positive else '-'} {point}  "
              f"Dice={dice(pred, gt):.4f}  ({time.time() - t0:.1f}s)")

    # --- Save result with the CT's spacing/origin/direction ---
    pred = target.numpy().copy()
    out_sitk = sitk.GetImageFromArray(pred)
    out_sitk.CopyInformation(image_sitk)
    sitk.WriteImage(out_sitk, str(out_path), useCompression=True)
    print(f"\nFinal Dice vs GT: {dice(pred, gt):.4f}")
    print(f"Saved prediction to {out_path}")


if __name__ == "__main__":
    main()
