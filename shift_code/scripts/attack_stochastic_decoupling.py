#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_PATH = "../models/sd2.1"

DEFAULT_PROMPTS_FILE = (
    "/home/ruibao/hf_cache/datasets/"
    "stable-diffusion-prompts-train.arrow"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stochastic trajectory decoupling attack (img2img + Euler a)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Local diffusers model path")
    parser.add_argument("--max_images", type=int, default=10, help="Number of images per algorithm to attack")
    parser.add_argument("--algorithms", default="TR,GS,RI,ROBIN,SFW,GM,PRC,WIND,SEAL", help="Comma-separated algorithm names")
    parser.add_argument("--input_dir", default="../outputs/smoke_test", help="Input directory of generated images")
    parser.add_argument("--output_dir", default="../outputs/attack", help="Output directory for attacked images")
    # parser.add_argument("--strengths", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7", help="Comma-separated strengths")
    parser.add_argument("--strengths", default="0.1", help="Comma-separated strengths")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion steps for img2img")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for img2img")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for attack sampling")
    parser.add_argument("--prompts_file", default=DEFAULT_PROMPTS_FILE, help="Local arrow file for prompts")
    parser.add_argument("--prompt_index", type=int, default=0, help="Prompt index used during generation")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda or cpu")
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
    if not snapshots_dir.is_dir():
        return model_path

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


def parse_strengths(raw: str) -> list[float]:
    strengths = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        strengths.append(float(item))
    return strengths


def extract_index(path: Path, algo: str) -> int | None:
    stem = path.stem  # e.g. TR_003
    prefix = f"{algo}_"
    if not stem.startswith(prefix):
        return None
    try:
        return int(stem[len(prefix):])
    except ValueError:
        return None


def load_prompts(prompts_file: Path, prompt_index: int, num_images: int) -> list[str]:
    if prompts_file.exists():
        dataset = Dataset.from_file(str(prompts_file))
    else:
        dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
    if prompt_index < 0 or prompt_index >= len(dataset):
        raise IndexError(f"prompt_index {prompt_index} out of range (0..{len(dataset)-1})")
    end_index = min(prompt_index + num_images, len(dataset))
    if end_index <= prompt_index:
        raise IndexError("num_images results in empty prompt range")
    return [dataset[i]["Prompt"] for i in range(prompt_index, end_index)]


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(device, args.dtype)

    model_path = find_working_snapshot(resolve_path(args.model_path))
    if model_path != resolve_path(args.model_path):
        print(f"[info] snapshot: {model_path}")

    prompts_file = resolve_path(args.prompts_file)
    prompts = load_prompts(prompts_file, args.prompt_index, args.max_images)
    print(f"Using aligned prompts from index {args.prompt_index}, count={len(prompts)}")

    strengths = [s for s in parse_strengths(args.strengths) if s > 0]
    
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(str(model_path), subfolder="scheduler")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(model_path),
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    input_root = resolve_path(args.input_dir)
    output_root = resolve_path(args.output_dir)

    

    for algo in algorithms:
        input_dir = input_root / algo
        if not input_dir.exists():
            print(f"[warn] input dir not found: {input_dir}")
            continue

        files = sorted(input_dir.glob(f"{algo}_*.png"))
        selected = []
        for path in files:
            idx = extract_index(path, algo)
            if idx is None or idx < 0:
                continue
            if idx >= args.max_images:
                continue
            selected.append((idx, path))

        if not selected:
            print(f"[warn] not found picture: {input_dir}")
            continue

        print(f"Attacking {algo}: {len(selected)} images, strengths={strengths}")

        for s_idx, strength in enumerate(strengths):
            strength_dir = output_root / algo / f"strength_{strength:.2f}"
            strength_dir.mkdir(parents=True, exist_ok=True)

            for idx, img_path in selected:
                image = Image.open(img_path).convert("RGB")
                if idx >= len(prompts):
                    raise IndexError(f"Prompt index {idx} out of range for loaded prompts (len={len(prompts)})")
                prompt = prompts[idx]
                generator = torch.Generator(device=device).manual_seed(args.seed + idx + s_idx * 10000)
                with torch.no_grad():
                    result = pipe(
                        prompt=prompt,
                        image=image,
                        strength=strength,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                    ).images[0]
                out_path = strength_dir / f"{algo}_{idx:03d}.png"
                result.save(out_path)
                if (idx + 1) % 10 == 0 or idx == 0:
                    print(f"[{algo}] strength={strength:.2f} saved {idx + 1}/{len(selected)} -> {out_path}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#python scripts/attack_stochastic_decoupling.py --max_images 10
