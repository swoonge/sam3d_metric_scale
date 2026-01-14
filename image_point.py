import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch


def resolve_sam2_root(repo_root: Path) -> Path:
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
    repo_root = Path(__file__).resolve().parent
    sam2_root = resolve_sam2_root(repo_root)
    default_checkpoint = sam2_root / "checkpoints" / "sam2.1_hiera_large.pt"
    default_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    default_image = sam2_root / "notebooks" / "videos" / "bedroom" / "00031.jpg"
    default_output_dir = repo_root / "outputs" / "sam2_masks"

    parser = argparse.ArgumentParser(description="SAM2 point-based mask UI")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--model-cfg", type=str, default=default_model_cfg)
    parser.add_argument("--image", type=Path, default=default_image)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Select device explicitly or use auto.",
    )
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    return parser.parse_args()


def autocast_context(device: str):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_help_panel(
    lines,
    font_scale=0.5,
    thickness=1,
    padding=10,
    line_gap=6,
    bg_color=(245, 245, 245),
    text_color=(20, 20, 20),
    border_color=(200, 200, 200),
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not lines:
        panel = np.full((60, 200, 3), bg_color, dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (199, 59), border_color, 1)
        return panel

    metrics = []
    max_w = 1
    total_h = 0
    for line in lines:
        (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        metrics.append((text_w, text_h, baseline))
        max_w = max(max_w, text_w)
        total_h += text_h + baseline

    if len(lines) > 1:
        total_h += line_gap * (len(lines) - 1)

    height = padding * 2 + max(total_h, 1)
    width = padding * 2 + max_w
    panel = np.full((height, width, 3), bg_color, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), border_color, 1)

    y = padding
    for line, (text_w, text_h, baseline) in zip(lines, metrics):
        y += text_h
        cv2.putText(
            panel,
            line,
            (padding, y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        y += baseline + line_gap

    return panel


def stack_with_panel_right(
    image: np.ndarray, panel: np.ndarray, bg_color=(245, 245, 245)
) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    panel_h, panel_w = panel.shape[:2]
    height = max(img_h, panel_h)
    canvas = np.full((height, img_w + panel_w, 3), bg_color, dtype=np.uint8)
    canvas[:img_h, :img_w] = image
    canvas[:panel_h, img_w:img_w + panel_w] = panel
    return canvas


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    repo_root = Path(__file__).resolve().parent
    sam2_root = resolve_sam2_root(repo_root)
    sam2_pkg_root = sam2_root / "sam2"

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    model_cfg_arg = args.model_cfg
    image_path = args.image
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model_cfg_path = Path(model_cfg_arg)
    if model_cfg_path.is_absolute():
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Missing model config: {model_cfg_path}")
        if sam2_pkg_root in model_cfg_path.parents:
            model_cfg = model_cfg_path.relative_to(sam2_pkg_root).as_posix()
        else:
            raise FileNotFoundError(
                "Model config must live under sam2 package configs; "
                f"got: {model_cfg_path}"
            )
    else:
        if not (sam2_pkg_root / model_cfg_path).exists():
            raise FileNotFoundError(
                f"Missing model config: {sam2_pkg_root / model_cfg_path}"
            )
        model_cfg = model_cfg_path.as_posix()
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    os.chdir(sam2_root)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(model_cfg, str(checkpoint_path), device=device)
    predictor = SAM2ImagePredictor(model)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with torch.inference_mode(), autocast_context(device):
        predictor.set_image(img_rgb)

    points = []  # list of (x,y)
    labels = []  # 1=pos, 0=neg
    last_mask = None
    last_score = None
    status_msg = ""
    status_time = 0.0

    def run_predict():
        nonlocal last_mask, last_score
        if len(points) == 0:
            last_mask = None
            last_score = None
            return

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        with torch.inference_mode(), autocast_context(device):
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        best = int(np.argmax(scores))
        last_mask = (masks[best] > 0).astype(np.uint8) * 255
        last_score = float(scores[best])

    def draw_ui():
        vis = img_bgr.copy()

        if last_mask is not None:
            green = np.zeros_like(vis)
            green[:, :, 1] = 255
            alpha = (last_mask > 0).astype(np.float32) * 0.35
            vis = (vis * (1 - alpha[..., None]) + green * alpha[..., None]).astype(np.uint8)

        for (x, y), lab in zip(points, labels):
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)

        panel = draw_help_panel()
        return stack_with_panel_right(vis, panel)

    def draw_help_panel():
        score_val = "-" if last_score is None else f"{last_score:.4f}"
        status_line = f"score: {score_val}  points: {len(points)}"
        lines = []
        if status_msg and (time.time() - status_time) < 5:
            lines.append(status_msg)
        lines.append(status_line)
        lines.append("LMB=positive")
        lines.append("Shift+LMB=negative")
        lines.append("z=undo")
        lines.append("r=reset")
        lines.append("s=save")
        lines.append("q/ESC=quit")
        return build_help_panel(lines)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                points.append((x, y))
                labels.append(0)
            else:
                points.append((x, y))
                labels.append(1)
            run_predict()

    cv2.namedWindow("sam2_point_ui", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("sam2_point_ui", args.window_width, args.window_height)
    cv2.setMouseCallback("sam2_point_ui", on_mouse)

    run_predict()

    save_index = 0
    while True:
        cv2.imshow("sam2_point_ui", draw_ui())
        k = cv2.waitKey(10) & 0xFF

        if k == ord("z"):
            if points:
                points.pop()
                labels.pop()
                run_predict()
        elif k == ord("r"):
            points.clear()
            labels.clear()
            last_mask = None
            last_score = None
        elif k == ord("s"):
            if last_mask is None:
                status_msg = "no mask to save"
                status_time = time.time()
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_mask = output_dir / f"{image_path.stem}_{save_index:03d}.png"
                cv2.imwrite(str(out_mask), last_mask)
                status_msg = f"saved: {out_mask.name}"
                status_time = time.time()
                save_index += 1
        elif k == ord("q") or k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
