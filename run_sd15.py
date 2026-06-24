"""Stable Diffusion v1.5 inference (base model)."""
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

PROMPT = "a studio ghibli style image of a girl in neon tokyo street at night, cinematic"
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, deformed, mutated, ugly, blurry, watermark, text, signature, extra limbs, missing limbs, worst quality, low quality, jpeg artifacts"

OUTPUT_FILE = "output.png"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    generator = torch.Generator(device=device).manual_seed(42)

    print("Generating image...")
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=512,
        height=512,
        generator=generator,
    ).images[0]

    image.save(OUTPUT_FILE)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
