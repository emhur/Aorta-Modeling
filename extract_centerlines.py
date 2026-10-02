"""Extract vessel centerlines from a binary mask (.nrrd / .nii.gz) with VMTK.

Must run in the `vmtk_x86` conda env (VMTK has no Apple Silicon build):
    conda run -n vmtk_x86 python extract_centerlines.py --mask AVT_dataset/Dongyang/D11/D11.seg.nrrd

Pipeline: mask -> marching-cubes surface (physical/LPS coords) -> Taubin smoothing
-> endpoints from mask skeleton -> vmtkcenterlines (pointlist seeds).
Source = endpoint with the largest inscribed radius (aortic root/inlet); targets = all
other endpoints. Override with --source / --targets (physical x y z, mm).
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vmtk import vmtkscripts


def largest_component(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def index_to_physical(img, ijk):
    """ijk: (N,3) in numpy (z,y,x) order -> (N,3) physical coords."""
    xyz = np.asarray(ijk, float)[:, ::-1]
    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    return origin + (direction @ (xyz * spacing).T).T


def mask_to_surface(img, mask):
    verts, faces, _, _ = measure.marching_cubes(np.pad(mask, 1).astype(np.uint8), 0.5)
    verts = index_to_physical(img, verts - 1)
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(verts, deep=True))
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(len(faces), numpy_to_vtkIdTypeArray(cells, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(ca)
    return poly


def skeleton_endpoints(img, mask):
    skel = morphology.skeletonize(mask)
    neighbors = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant")
    ends = np.argwhere(skel & (neighbors == 2))  # self + exactly one neighbour
    radius = ndimage.distance_transform_edt(mask, sampling=img.GetSpacing()[::-1])
    return index_to_physical(img, ends), radius[tuple(ends.T)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="Output .vtp (default: <mask stem>.centerlines.vtp)")
    p.add_argument("--source", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--targets", type=float, nargs="+", default=None, help="x y z [x y z ...]")
    p.add_argument("--min-branch-mm", type=float, default=10.0,
                   help="Drop auto-detected endpoints closer than this to another endpoint")
    p.add_argument("--smooth-iters", type=int, default=30)
    p.add_argument("--resampling", type=float, default=0.5, help="Centerline point spacing (mm)")
    args = p.parse_args()

    img = sitk.ReadImage(str(args.mask))
    mask = largest_component(sitk.GetArrayFromImage(img) > 0)
    print(f"Mask voxels: {mask.sum()}  spacing: {img.GetSpacing()}")

    surface = mask_to_surface(img, mask)
    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = surface
    smoother.Method = "taubin"
    smoother.NumberOfIterations = args.smooth_iters
    smoother.PassBand = 0.1
    smoother.Execute()
    surface = smoother.Surface

    if args.source is not None and args.targets is not None:
        source = np.array(args.source)
        targets = np.array(args.targets).reshape(-1, 3)
    else:
        ends, radii = skeleton_endpoints(img, mask)
        if len(ends) < 2:
            raise SystemExit("Found <2 skeleton endpoints; pass --source and --targets manually.")
        order = np.argsort(-radii)
        keep = []
        for i in order:  # greedily drop spurious endpoints that cluster together
            if all(np.linalg.norm(ends[i] - ends[j]) >= args.min_branch_mm for j in keep):
                keep.append(i)
        source, targets = ends[keep[0]], ends[keep[1:]]
        print(f"Auto source (r={radii[keep[0]]:.1f} mm): {np.round(source, 1)}")
    print(f"{len(targets)} target(s):\n{np.round(targets, 1)}")

    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = source.tolist()
    cl.TargetPoints = targets.ravel().tolist()
    cl.AppendEndPoints = 1
    cl.Resampling = 1
    cl.ResamplingStepLength = args.resampling
    cl.Execute()

    out = args.out or args.mask.with_name(args.mask.name.split(".")[0] + ".centerlines.vtp")
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Surface = cl.Centerlines
    writer.OutputFileName = str(out)
    writer.Execute()

    surf_out = out.with_name(out.name.replace(".centerlines.vtp", ".surface.vtp"))
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Surface = surface
    writer.OutputFileName = str(surf_out)
    writer.Execute()
    print(f"Centerlines: {cl.Centerlines.GetNumberOfCells()} lines, "
          f"{cl.Centerlines.GetNumberOfPoints()} points -> {out}\nSurface -> {surf_out}")


if __name__ == "__main__":
    main()
