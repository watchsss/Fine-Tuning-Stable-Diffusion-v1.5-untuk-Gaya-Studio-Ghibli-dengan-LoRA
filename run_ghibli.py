"""Stable Diffusion v1.5 + Ghibli LoRA inference."""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_ID = "stabilityai/sd-vae-ft-mse"
LORA_PATH = "d:/FinetuningAI/ghibli-lora"

PROMPT = "a studio ghibli style image of a girl in neon tokyo street at night, cinematic"
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, deformed, mutated, ugly, blurry, watermark, text, signature, extra limbs, missing limbs, worst quality, low quality, jpeg artifacts"

LORA_SCALE = 0.9

OUTPUT_FILE = "ghibli_output.png"


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

    pipe.load_lora_weights(LORA_PATH)
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=device).manual_seed(42)

    print("Generating image...")
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=25,
        guidance_scale=7.5,
        width=512,
        height=512,
        generator=generator,
        cross_attention_kwargs={"scale": LORA_SCALE},
    ).images[0]

    image.save(OUTPUT_FILE)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
