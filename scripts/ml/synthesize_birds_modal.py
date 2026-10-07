"""Make paired bird/no_bird crops of a seed image with Qwen-Image 2.1 on Modal."""

import io
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from PIL import Image

    from birdspotter.ml.synthetic import CropPlan

QWEN_MODEL = "Qwen/Qwen-Image-2.1"
# QwenImage21Pipeline is only on diffusers main; pin the commit that was tested against.
DIFFUSERS_COMMIT = "8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd"
OUTPUT_SIZE = 640

app = modal.App("birdspotter-synthetic-birds")
hf_cache = modal.Volume.from_name("birdspotter-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        "transformers==5.18.0",
        "accelerate",
        "pillow==12.3.0",
        f"diffusers @ git+https://github.com/huggingface/diffusers@{DIFFUSERS_COMMIT}",
    )
    .env({"HF_HOME": "/mnt/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
)

with image.imports():
    import numpy as np
    import torch
    from diffusers import QwenImage21Pipeline  # ty: ignore[unresolved-import] - Modal image only
    from PIL import Image


@app.cls(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/mnt/hf": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=10,
    scaledown_window=120,
)
class BirdEditor:
    @modal.enter()
    def load(self) -> None:
        self.pipe = QwenImage21Pipeline.from_pretrained(QWEN_MODEL, dtype=torch.bfloat16).to("cuda")
        hf_cache.commit()

    def vae_roundtrip(self, image: "Image.Image") -> "Image.Image":
        """Encode and decode without diffusion, so negatives carry the VAE's artefacts too."""
        rgb = np.asarray(image, dtype=np.float32) / 127.5 - 1
        # The Qwen-Image 2.1 VAE takes RGBA video tensors (B, C, T, H, W).
        rgba = np.concatenate([rgb, np.ones_like(rgb[..., :1])], axis=-1)
        x = torch.from_numpy(rgba).permute(2, 0, 1)[None, :, None].to("cuda", self.pipe.vae.dtype)
        with torch.no_grad():
            decoded = self.pipe.vae.decode(self.pipe.vae.encode(x).latent_dist.mode()).sample
        out = ((decoded[0, :3, 0].float().clamp(-1, 1) + 1) * 127.5).round().byte()
        return Image.fromarray(out.permute(1, 2, 0).cpu().numpy())

    @modal.method()
    def edit(self, crop_png: bytes, prompt: str, seed: int, steps: int) -> tuple[bytes, bytes]:
        """Return the bird-added edit and the VAE round trip of the unedited crop."""
        crop = Image.open(io.BytesIO(crop_png)).convert("RGB")
        negative = self.vae_roundtrip(
            crop.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)
        )
        edited = self.pipe(
            prompt,
            image=crop,
            height=OUTPUT_SIZE,
            width=OUTPUT_SIZE,
            output_resolution=OUTPUT_SIZE,
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        return png_bytes(edited), png_bytes(negative)


def png_bytes(image: "Image.Image") -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@dataclass
class SeedJob:
    """One seed image's crop plans and the run directory its pairs are written to."""

    seed_path: Path
    destination: Path
    seed: int
    plans: list["CropPlan"]
    crops: list["Image.Image"]


def prepare_job(  # noqa: PLR0913
    seed_path: Path,
    destination: Path,
    *,
    count: int,
    min_fraction: float,
    max_fraction: float,
    seed: int,
) -> SeedJob:
    from PIL import Image  # noqa: PLC0415

    from birdspotter.ml.synthetic import plan_crops  # noqa: PLC0415

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"{destination} is not empty; delete it or pass --output")
    for label in ("bird", "no_bird"):
        (destination / label).mkdir(parents=True, exist_ok=True)
    source = Image.open(seed_path).convert("RGB")
    plans = plan_crops(
        *source.size,
        count=count,
        min_fraction=min_fraction,
        max_fraction=max_fraction,
        seed=seed,
    )
    # Seed images contain no birds, so each crop (after a VAE round trip) is the no_bird half.
    crops = [
        source.crop((plan.x, plan.y, plan.x + plan.size, plan.y + plan.size)) for plan in plans
    ]
    return SeedJob(seed_path, destination, seed, plans, crops)


def write_job(job: SeedJob, pairs: list[tuple[bytes, bytes]], config: dict[str, object]) -> None:
    """Write a seed's pairs and manifest, then its build config to mark the run complete."""
    from PIL import Image  # noqa: PLC0415

    with (job.destination / "manifest.jsonl").open("w") as manifest:
        for plan, (bird, no_bird) in zip(job.plans, pairs, strict=True):
            for label, data in (("no_bird", no_bird), ("bird", bird)):
                image = Image.open(io.BytesIO(data))
                path = job.destination / label / f"{plan.index:06d}.png"
                image.convert("RGB").resize(
                    (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS
                ).save(path)
                record = plan.to_json() | {
                    "label": label,
                    "path": str(path.relative_to(job.destination)),
                }
                manifest.write(json.dumps(record) + "\n")
    config = config | {
        "created_at": datetime.now(UTC).isoformat(),
        "seed_image": str(job.seed_path),
        "seed": job.seed,
    }
    (job.destination / "build-config.json").write_text(json.dumps(config, indent=2) + "\n")


@app.local_entrypoint()
def main(  # noqa: PLR0913, PLR0917
    seed_image: str = "",
    seed_list: str = "",
    output: str = "",
    count: int = 100,
    min_fraction: float = 0.3,
    max_fraction: float = 1.0,
    seed: int = 0,
    steps: int = 40,
) -> None:
    """Edit one seed image, or every image in --seed-list in a single app run.

    With --seed-list (one image path per line), line i uses seed ``seed + i`` and writes to
    ``<output>/<image's parent dir>-n<count>-s<seed + i>``. Runs that already have a
    build-config.json are skipped, so an interrupted list can be resumed.
    """
    if seed_list:
        root = Path(output) if output else Path("data/synthetic") / Path(seed_list).stem
        entries = [
            (Path(line), root / f"{Path(line).parent.name}-n{count}-s{seed + i}", seed + i)
            for i, line in enumerate(Path(seed_list).read_text().split())
        ]
        entries = [e for e in entries if not (e[1] / "build-config.json").exists()]
        for _, destination, _ in entries:
            shutil.rmtree(destination, ignore_errors=True)  # partial output of a killed run
    else:
        seed_path = (
            Path(seed_image) if seed_image else max(Path("data/beelink").glob("camera-*.jpg"))
        )
        run_name = f"{seed_path.stem}-n{count}-s{seed}"
        entries = [(seed_path, Path(output) if output else Path("data/synthetic") / run_name, seed)]
    jobs = [
        prepare_job(
            path,
            destination,
            count=count,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            seed=job_seed,
        )
        for path, destination, job_seed in entries
    ]
    config: dict[str, object] = {
        "model": QWEN_MODEL,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "count": count,
        "min_fraction": min_fraction,
        "max_fraction": max_fraction,
        "steps": steps,
        "output_size": OUTPUT_SIZE,
    }
    # One starmap over every crop keeps the containers warm across seed images.
    pairs = iter(
        BirdEditor().edit.starmap(
            [
                (png_bytes(crop), plan.bird_prompt, plan.seed, steps)
                for job in jobs
                for plan, crop in zip(job.plans, job.crops, strict=True)
            ]
        )
    )
    for done, job in enumerate(jobs, 1):
        write_job(job, [next(pairs) for _ in job.plans], config)
        print(f"[{done}/{len(jobs)}] Wrote {count} bird/no_bird pairs to {job.destination}")
