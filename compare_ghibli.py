"""Side-by-side comparison: base SD1.5 vs SD1.5 + Ghibli LoRA.

All generation parameters (prompt, seed, sampler, VAE, steps) are identical;
the only difference is whether the LoRA weights are applied.
"""
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_ID = "stabilityai/sd-vae-ft-mse"
LORA_PATH = "d:/FinetuningAI/ghibli-lora"

PROMPT = "a studio ghibli style image of a beautiful mountain"
NEGATIVE_PROMPT = "lowres, bad anatomy, deformed, ugly, blurry, watermark, text, worst quality, low quality"

SEED = 123
STEPS = 25
GUIDANCE = 7.5
LORA_SCALE = 0.9
OUTPUT_FILE = "comparison.png"


def generate(pipe, device, lora_scale=None):
    generator = torch.Generator(device=device).manual_seed(SEED)
    kwargs = {}
    if lora_scale is not None:
        kwargs["cross_attention_kwargs"] = {"scale": lora_scale}
    return pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        width=512,
        height=512,
        generator=generator,
        **kwargs,
    ).images[0]


def label(img, text):
    """Add a caption bar above the image."""
    bar_h = 36
    out = Image.new("RGB", (img.width, img.height + bar_h), "black")
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 7), text, fill="white", font=font)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=dtype, safety_checker=None
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe.enable_attention_slicing()

    # Base output is generated before the LoRA weights are loaded.
    print("Generating base SD1.5...")
    img_base = generate(pipe, device, lora_scale=None)

    pipe.load_lora_weights(LORA_PATH)
    print("Generating SD1.5 + Ghibli LoRA...")
    img_lora = generate(pipe, device, lora_scale=LORA_SCALE)

    a = label(img_base, "BASE SD1.5")
    b = label(img_lora, "+ LoRA Ghibli")
    combined = Image.new("RGB", (a.width + b.width, a.height), "black")
    combined.paste(a, (0, 0))
    combined.paste(b, (a.width, 0))
    combined.save(OUTPUT_FILE)
    img_base.save("compare_base.png")
    img_lora.save("compare_lora.png")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
