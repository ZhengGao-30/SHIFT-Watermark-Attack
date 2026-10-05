#!/usr/bin/env python3
"""
Batch watermark removal attack adapted from the paper's run_imprint_removal.py.

Goal:
- take already-generated images from an input directory
- run removal attack directly on those images
- save only attacked images
- support fair comparison through parameter grids

Expected input layout (recommended):
    input_dir/
        TR/TR_000.png
        TR/TR_001.png
        GS/GS_000.png
        ...

It also supports nested layouts and will preserve the relative subdirectory structure.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import imprint_utils


DEFAULT_MODEL_PATH = "/srv/scratch/z5532117/models/sd2.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch removal attack on existing images with parameter-grid support."
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Local diffusers model path for the attacker model")
    parser.add_argument("--input_dir", type=str, default="/srv/scratch/z5532117/outputs/smoke_test",
                        help="Input root directory containing images")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5532117/outputs/black_box_attack",
                        help="Output root directory for attacked images")

    parser.add_argument("--algorithms", type=str, default="",
                        help="Comma-separated algorithm names to keep, e.g. TR,GS,RI. Empty means infer all.")
    parser.add_argument("--max_images", type=int, default=-1,
                        help="Maximum images per algorithm/subfolder. -1 means no limit.")
    parser.add_argument("--glob", type=str, default="*.png",
                        help="Image glob pattern inside each selected folder")

    # grid parameters
    parser.add_argument("--steps_list", type=str, default="151",
                        help="Comma-separated optimization steps, e.g. 50,100,151,200")
    parser.add_argument("--lr_list", type=str, default="0.01",
                        help="Comma-separated learning rates, e.g. 0.001,0.005,0.01")
    parser.add_argument("--inv_steps_list", type=str, default="50",
                        help="Comma-separated inversion steps, e.g. 20,50,100")
    parser.add_argument("--guidance_scale_list", type=str, default="1.0",
                        help="Comma-separated guidance scales used in diffpipe optimization")
    parser.add_argument("--scheduler_list", type=str, default="DDIM",
                        help="Comma-separated attacker schedulers, e.g. DDIM,PNDM")

    parser.add_argument("--grid_mode", choices=["full", "paired"], default="paired",
                        help="full = Cartesian product, paired = zip lists by index (all lists must have same length or length 1)")
    parser.add_argument("--tag_mode", choices=["compact", "verbose"], default="compact",
                        help="Folder naming style for each parameter setting")

    parser.add_argument("--resolution", type=int, default=512,
                        help="Resize input images to this square resolution before attack")
    parser.add_argument("--save_every", type=int, default=0,
                        help="If >0, also save intermediate images every N steps")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing final outputs")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip image if final output already exists")
    parser.add_argument("--seed", type=int, default=1,
                        help="Base seed")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


def find_working_snapshot(model_path: Path) -> Path:
    model_index = model_path / "model_index.json"
    if model_index.is_file():
        return model_path

    snapshots_dir = model_path.parent if model_path.name else model_path
    if snapshots_dir.name != "snapshots":
        candidate = model_path / "snapshots"
        if candidate.is_dir():
            snapshots_dir = candidate

    if snapshots_dir.is_dir():
        for snapshot in sorted(snapshots_dir.iterdir()):
            if snapshot.is_dir() and (snapshot / "model_index.json").is_file():
                return snapshot

    return model_path


def resolve_dtype(device: str, dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    return torch.float16 if device == "cuda" else torch.float32


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_list(raw: str, cast):
    items = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        items.append(cast(x))
    return items


def build_settings(args: argparse.Namespace):
    steps_list = parse_list(args.steps_list, int)
    lr_list = parse_list(args.lr_list, float)
    inv_steps_list = parse_list(args.inv_steps_list, int)
    guidance_list = parse_list(args.guidance_scale_list, float)
    scheduler_list = parse_list(args.scheduler_list, str)

    if args.grid_mode == "full":
        combos = list(itertools.product(
            scheduler_list, inv_steps_list, steps_list, lr_list, guidance_list
        ))
    else:
        lists = [scheduler_list, inv_steps_list, steps_list, lr_list, guidance_list]
        lengths = [len(x) for x in lists]
        max_len = max(lengths)
        for n in lengths:
            if n not in (1, max_len):
                raise ValueError(
                    f"paired mode requires each list length to be 1 or {max_len}, got {lengths}"
                )

        def get(lst, i):
            return lst[0] if len(lst) == 1 else lst[i]

        combos = [
            (
                get(scheduler_list, i),
                get(inv_steps_list, i),
                get(steps_list, i),
                get(lr_list, i),
                get(guidance_list, i),
            )
            for i in range(max_len)
        ]

    settings = []
    for scheduler, inv_steps, steps, lr, guidance_scale in combos:
        settings.append({
            "scheduler_attacker": scheduler,
            "num_inference_steps_attacker": inv_steps,
            "steps": steps,
            "lr": lr,
            "guidance_scale": guidance_scale,
        })
    return settings


def format_setting_tag(setting: dict, mode: str) -> str:
    scheduler = setting["scheduler_attacker"]
    inv_steps = setting["num_inference_steps_attacker"]
    steps = setting["steps"]
    lr = setting["lr"]
    guidance = setting["guidance_scale"]

    if mode == "verbose":
        return (
            f"scheduler_{scheduler}"
            f"__inv_{inv_steps}"
            f"__steps_{steps}"
            f"__lr_{lr:g}"
            f"__guidance_{guidance:g}"
        )

    return f"{scheduler}_inv{inv_steps}_st{steps}_lr{lr:g}_g{guidance:g}"


def infer_algorithm_from_path(path: Path, input_root: Path, requested_algorithms: set[str]) -> str:
    rel_parts = path.relative_to(input_root).parts
    if requested_algorithms:
        for part in rel_parts:
            if part in requested_algorithms:
                return part
    # fallback: use first directory under input root, otherwise file stem prefix
    if len(rel_parts) >= 2:
        return rel_parts[0]
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return "unknown"


def collect_images(input_root: Path, glob_pattern: str, algorithms: set[str], max_images: int):
    all_paths = sorted(input_root.rglob(glob_pattern))
    grouped = {}

    for path in all_paths:
        if not path.is_file():
            continue
        algo = infer_algorithm_from_path(path, input_root, algorithms)
        if algorithms and algo not in algorithms:
            continue
        grouped.setdefault(algo, []).append(path)

    selected = {}
    for algo, paths in grouped.items():
        selected[algo] = paths if max_images < 0 else paths[:max_images]
    return selected


def load_pipe(model_path: Path, scheduler_name: str, device: str):
    pipe_attacker, _, inverse_scheduler = imprint_utils.load_attacker_pipe(
        modelid=str(model_path),
        scheduler=scheduler_name,
        device=torch.device(device),
    )
    diffpipe = imprint_utils.DiffPipe(
        pipe_attacker,
        scheduler=inverse_scheduler,
        device=pipe_attacker.device,
    )
    return pipe_attacker, inverse_scheduler, diffpipe


def preprocess_image(path: Path, resolution: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution))
    return image


def attack_one_image(
    image: Image.Image,
    pipe_attacker,
    inverse_scheduler,
    diffpipe,
    inv_steps: int,
    steps: int,
    lr: float,
    guidance_scale: float,
    save_every: int,
    intermediate_prefix: Path | None = None,
):
    z0_original = imprint_utils.pixel_to_latent(image, pipe_attacker)
    z0 = torch.nn.Parameter(z0_original.detach().clone())
    optim = torch.optim.Adam([z0], lr=lr)

    image_pt = to_tensor(image).unsqueeze(0).to(pipe_attacker.device).to(dtype=torch.float32)

    with torch.no_grad():
        zT_retrieved = imprint_utils.invert_image(
            pipe=pipe_attacker,
            image_pt=image_pt,
            scheduler=inverse_scheduler,
            num_inference_steps=inv_steps,
        )
        zT_target = zT_retrieved.detach() * -1.0

    for step in range(steps):
        optim.zero_grad()
        inverted_latent = diffpipe(z0, "", guidance_scale=guidance_scale)
        loss = torch.nn.functional.mse_loss(inverted_latent, zT_target)
        loss.backward()
        optim.step()

        if save_every > 0 and intermediate_prefix is not None and step > 0 and step % save_every == 0:
            inter_img = imprint_utils.latent_to_pil(z0, pipe_attacker)[0]
            inter_img.save(str(intermediate_prefix.with_name(f"{intermediate_prefix.stem}_step{step:04d}.png")))

    final_img = imprint_utils.latent_to_pil(z0, pipe_attacker)[0]
    return final_img


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(device, args.dtype)
    model_path = find_working_snapshot(resolve_path(args.model_path))
    input_root = resolve_path(args.input_dir)
    output_root = resolve_path(args.output_dir)

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    requested_algorithms = {x.strip() for x in args.algorithms.split(",") if x.strip()}
    grouped_images = collect_images(input_root, args.glob, requested_algorithms, args.max_images)
    if not grouped_images:
        raise RuntimeError(f"No images found under {input_root} with glob={args.glob}")

    settings = build_settings(args)

    manifest = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "model_path": str(model_path),
        "device": device,
        "dtype": str(dtype),
        "grid_mode": args.grid_mode,
        "settings": settings,
        "algorithms": sorted(grouped_images.keys()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "attack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[info] device={device}, dtype={dtype}, model_path={model_path}")
    print(f"[info] found algorithms={sorted(grouped_images.keys())}")
    print(f"[info] total settings={len(settings)}")

    loaded = {}
    for setting in settings:
        scheduler_name = setting["scheduler_attacker"]
        if scheduler_name not in loaded:
            print(f"[info] loading attacker pipe for scheduler={scheduler_name}")
            loaded[scheduler_name] = load_pipe(model_path, scheduler_name, device)

    for algo, img_paths in grouped_images.items():
        print(f"[info] attacking algo={algo}, num_images={len(img_paths)}")

        for setting_idx, setting in enumerate(settings):
            tag = format_setting_tag(setting, args.tag_mode)
            setting_dir = output_root / algo / tag
            setting_dir.mkdir(parents=True, exist_ok=True)

            pipe_attacker, inverse_scheduler, diffpipe = loaded[setting["scheduler_attacker"]]

            for img_idx, img_path in enumerate(tqdm.tqdm(img_paths, desc=f"{algo} | {tag}")):
                rel = img_path.relative_to(input_root)
                # keep nested structure under each setting, but avoid duplicating algo twice
                tail_parts = rel.parts[1:] if len(rel.parts) > 1 and rel.parts[0] == algo else rel.parts
                out_path = setting_dir.joinpath(*tail_parts)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if out_path.exists() and args.skip_existing and not args.overwrite:
                    continue

                image = preprocess_image(img_path, args.resolution)

                # deterministic but distinct seed per image and setting
                local_seed = args.seed + setting_idx * 100000 + img_idx
                set_seed(local_seed)

                intermediate_prefix = out_path.with_suffix("") if args.save_every > 0 else None
                final_img = attack_one_image(
                    image=image,
                    pipe_attacker=pipe_attacker,
                    inverse_scheduler=inverse_scheduler,
                    diffpipe=diffpipe,
                    inv_steps=setting["num_inference_steps_attacker"],
                    steps=setting["steps"],
                    lr=setting["lr"],
                    guidance_scale=setting["guidance_scale"],
                    save_every=args.save_every,
                    intermediate_prefix=intermediate_prefix,
                )
                final_img.save(out_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[done] outputs saved to: {output_root}")


if __name__ == "__main__":
    main()
