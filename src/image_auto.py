"""SAM2 자동 마스크 생성기.

- 입력: 단일 RGB 이미지 (옵션: depth 이미지)
- 출력: 자동 생성된 마스크 PNG들과 시각화 이미지
"""

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch


def resolve_sam2_root(repo_root: Path) -> Path:
    """SAM2 레포 위치를 환경변수/기본 후보에서 탐색."""
    env_root = os.environ.get("SAM2_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(repo_root / "sam2")
    candidates.append(repo_root.parent / "sam2")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else repo_root / "sam2"


def parse_args() -> argparse.Namespace:
    """CLI 인자 정의."""
    repo_root = Path(__file__).resolve().parents[1]
    sam2_root = resolve_sam2_root(repo_root)
    default_checkpoint = sam2_root / "checkpoints" / "sam2.1_hiera_large.pt"
    default_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    default_image = sam2_root / "notebooks" / "videos" / "bedroom" / "00031.jpg"
    default_output_dir = repo_root / "outputs" / "sam2_masks"

    parser = argparse.ArgumentParser(description="SAM2 automatic mask generation")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--model-cfg", type=str, default=default_model_cfg)
    parser.add_argument("--image", type=Path, default=default_image)
    parser.add_argument("--depth-image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Select device explicitly or use auto.",
    )
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.9)
    parser.add_argument("--min-mask-region-area", type=int, default=500)
    parser.add_argument(
        "--depth-same-surface-thresh",
        type=float,
        default=None,
        help="Depth threshold for nested-surface filtering. If omitted, auto infer.",
    )
    parser.add_argument(
        "--dedupe-iou-thresh",
        type=float,
        default=0.85,
        help="IoU threshold for duplicate mask removal.",
    )
    parser.add_argument(
        "--nested-containment-thresh",
        type=float,
        default=0.98,
        help="Containment ratio threshold for nested mask pruning.",
    )
    parser.add_argument(
        "--depth-connected-ratio-thresh",
        type=float,
        default=0.9,
        help="Minimum ratio of depth-connected pixels for nested surface pruning.",
    )
    parser.add_argument(
        "--keep-largest-component",
        dest="keep_largest_component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the largest connected component per mask.",
    )
    parser.add_argument(
        "--filter-border",
        dest="filter_border",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove masks that touch image top or bottom border.",
    )
    parser.add_argument(
        "--apply-extra-filtering",
        dest="apply_extra_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply additional filtering pipeline (dedupe/border/depth-nested).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clean-output",
        dest="clean_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete existing files in output dir before saving (default: enabled).",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    """상대 경로를 실행 디렉터리 기준 절대 경로로 변환."""
    if path.is_absolute():
        return path
    return (base / path).resolve()


def autocast_context(device: str):
    """CUDA일 때만 bfloat16 autocast를 활성화."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def clear_output_files(output_dir: Path) -> None:
    """출력 디렉터리의 기존 파일을 제거한다."""
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_file():
            child.unlink()


def resolve_model_cfg(model_cfg_arg: str, sam2_pkg_root: Path) -> str:
    """sam2 패키지 루트 기준 model config 경로를 정규화."""
    model_cfg_path = Path(model_cfg_arg)
    if model_cfg_path.is_absolute():
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Missing model config: {model_cfg_path}")
        if sam2_pkg_root in model_cfg_path.parents:
            return model_cfg_path.relative_to(sam2_pkg_root).as_posix()
        raise FileNotFoundError(
            "Model config must live under sam2 package configs; "
            f"got: {model_cfg_path}"
        )
    if not (sam2_pkg_root / model_cfg_path).exists():
        raise FileNotFoundError(f"Missing model config: {sam2_pkg_root / model_cfg_path}")
    return model_cfg_path.as_posix()


def infer_depth_surface_threshold(depth_map: np.ndarray) -> tuple[float, float]:
    """유효 depth 분포를 보고 same-surface 임계값을 자동 추정."""
    valid_depth = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
    if valid_depth.size == 0:
        raise ValueError("No valid depth values found.")

    depth_median_global = float(np.median(valid_depth))
    if depth_median_global > 10.0:
        depth_same_surface_thresh = 20.0  # likely millimeter scale
    else:
        depth_same_surface_thresh = 0.02  # likely meter scale

    return depth_median_global, depth_same_surface_thresh


def get_mask_depth_values(mask: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    vals = depth_map[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return vals


def compute_mask_info(mask: np.ndarray, depth_map: np.ndarray | None) -> dict | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None

    info = {
        "mask": mask,
        "area": int(mask.sum()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
    }

    if depth_map is not None:
        depth_vals = get_mask_depth_values(mask, depth_map)
        if depth_vals.size == 0:
            return None
        info.update(
            {
                "depth_median": float(np.median(depth_vals)),
                "depth_mean": float(np.mean(depth_vals)),
                "depth_std": float(np.std(depth_vals)),
            }
        )

    return info


def is_border_mask(info: dict, height: int) -> bool:
    return info["y_min"] == 0 or info["y_max"] == height - 1


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return labels == largest_label


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def remove_duplicate_masks(mask_infos: list[dict], iou_thresh: float) -> list[dict]:
    keep = [True] * len(mask_infos)
    for i in range(len(mask_infos)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(mask_infos)):
            if not keep[j]:
                continue
            iou = mask_iou(mask_infos[i]["mask"], mask_infos[j]["mask"])
            if iou <= iou_thresh:
                continue
            if mask_infos[i]["area"] >= mask_infos[j]["area"]:
                keep[j] = False
            else:
                keep[i] = False
                break
    return [m for k, m in zip(keep, mask_infos) if k]


def bbox_contains(small: dict, large: dict) -> bool:
    return (
        small["x_min"] >= large["x_min"]
        and small["x_max"] <= large["x_max"]
        and small["y_min"] >= large["y_min"]
        and small["y_max"] <= large["y_max"]
    )


def mask_containment_ratio(small_mask: np.ndarray, large_mask: np.ndarray) -> float:
    inter = np.logical_and(small_mask, large_mask).sum()
    area_small = small_mask.sum()
    if area_small == 0:
        return 0.0
    return float(inter / area_small)


def depth_surface_connected(
    small_mask: np.ndarray,
    large_mask: np.ndarray,
    depth_map: np.ndarray,
    depth_thresh: float,
    connected_ratio_thresh: float,
) -> bool:
    ys, xs = np.where(small_mask)
    total = len(xs)
    if total == 0:
        return False

    connected = 0
    for y, x in zip(ys, xs):
        d = depth_map[y, x]
        if not np.isfinite(d) or d <= 0:
            continue

        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= depth_map.shape[0]:
                    continue
                if nx < 0 or nx >= depth_map.shape[1]:
                    continue
                if not large_mask[ny, nx]:
                    continue
                d2 = depth_map[ny, nx]
                if np.isfinite(d2) and abs(d - d2) < depth_thresh:
                    found = True
                    break
            if found:
                break

        if found:
            connected += 1

    ratio = connected / total
    return ratio > connected_ratio_thresh


def remove_nested_same_depth_masks(
    mask_infos: list[dict],
    depth_map: np.ndarray,
    depth_thresh: float,
    containment_thresh: float,
    connected_ratio_thresh: float,
) -> list[dict]:
    keep = [True] * len(mask_infos)
    order = np.argsort([m["area"] for m in mask_infos])

    for oi in order:
        if not keep[oi]:
            continue
        small = mask_infos[oi]

        for oj in range(len(mask_infos)):
            if oi == oj or not keep[oj]:
                continue
            large = mask_infos[oj]

            if large["area"] <= small["area"] * 1.2:
                continue
            if not bbox_contains(small, large):
                continue

            contain_ratio = mask_containment_ratio(small["mask"], large["mask"])
            if contain_ratio < containment_thresh:
                continue

            if depth_surface_connected(
                small["mask"],
                large["mask"],
                depth_map,
                depth_thresh,
                connected_ratio_thresh,
            ):
                print(
                    f"[REMOVE] nested connected surface | small area={small['area']} "
                    f"large area={large['area']} contain={contain_ratio:.3f}"
                )
                keep[oi] = False
                break

    return [m for k, m in zip(keep, mask_infos) if k]


def main() -> None:
    """자동 마스크 생성 후 시각화/마스크 파일 저장."""
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    repo_root = Path(__file__).resolve().parents[1]
    sam2_root = resolve_sam2_root(repo_root)
    sam2_pkg_root = sam2_root / "sam2"

    checkpoint_path = resolve_path(args.checkpoint, Path.cwd())
    image_path = resolve_path(args.image, Path.cwd())
    depth_path = (
        resolve_path(args.depth_image, Path.cwd())
        if args.depth_image is not None
        else None
    )
    output_dir = resolve_path(args.output_dir, Path.cwd())
    model_cfg = resolve_model_cfg(args.model_cfg, sam2_pkg_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if depth_path is not None and not depth_path.exists():
        raise FileNotFoundError(f"Missing depth image: {depth_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clear_output_files(output_dir)

    # sam2 패키지를 import 가능하도록 경로 등록
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    os.chdir(sam2_root)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_h, _ = img_rgb.shape[:2]

    depth_map = None
    depth_same_surface_thresh = None
    if depth_path is not None:
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)
        depth_map = depth_raw.astype(np.float32)
        if args.depth_same_surface_thresh is None:
            depth_median_global, depth_same_surface_thresh = infer_depth_surface_threshold(
                depth_map
            )
            print(f"[INFO] global median depth: {depth_median_global:.4f}")
            print(
                "[INFO] same-surface depth threshold: "
                f"{depth_same_surface_thresh}"
            )
        else:
            depth_same_surface_thresh = args.depth_same_surface_thresh
            print(
                "[INFO] same-surface depth threshold (manual): "
                f"{depth_same_surface_thresh}"
            )

    model = build_sam2(model_cfg, str(checkpoint_path), device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )

    print("Generating masks...")
    with torch.inference_mode(), autocast_context(device):
        masks = mask_generator.generate(img_rgb)
    print("Total masks:", len(masks))

    final_masks: list[np.ndarray]
    if args.apply_extra_filtering:
        mask_infos = []
        for mask_item in masks:
            mask = np.asarray(mask_item["segmentation"], dtype=bool)
            if args.keep_largest_component:
                mask = keep_largest_component(mask)
            info = compute_mask_info(mask, depth_map)
            if info is None:
                continue
            mask_infos.append(info)

        print("Valid masks after preprocess:", len(mask_infos))
        mask_infos = remove_duplicate_masks(mask_infos, iou_thresh=args.dedupe_iou_thresh)
        print("After duplicate filter:", len(mask_infos))

        if args.filter_border:
            mask_infos = [m for m in mask_infos if not is_border_mask(m, image_h)]
            print("After border filter:", len(mask_infos))

        if depth_map is not None:
            mask_infos = remove_nested_same_depth_masks(
                mask_infos,
                depth_map=depth_map,
                depth_thresh=float(depth_same_surface_thresh),
                containment_thresh=args.nested_containment_thresh,
                connected_ratio_thresh=args.depth_connected_ratio_thresh,
            )
            print("After nested depth filter:", len(mask_infos))

        final_masks = [m["mask"] for m in mask_infos]
    else:
        final_masks = [np.asarray(m["segmentation"], dtype=bool) for m in masks]

    vis = img_bgr.copy()
    rng = np.random.default_rng(args.seed)
    for mask in final_masks:
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        overlay = (
            vis[mask].astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5
        )
        vis[mask] = overlay.astype(np.uint8)

    # NOTE: Keep visualization filename outside "<stem>_*.png" mask glob pattern.
    vis_path = output_dir / f"vis_{image_path.stem}.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"saved visualization: {vis_path}")

    for i, mask in enumerate(final_masks):
        mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
        mask_path = output_dir / f"{image_path.stem}_{i:03d}.png"
        cv2.imwrite(str(mask_path), mask_u8)
        print(f"saved: {mask_path}")


if __name__ == "__main__":
    main()
