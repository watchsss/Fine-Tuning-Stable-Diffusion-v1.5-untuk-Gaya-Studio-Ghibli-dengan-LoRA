# Train a Ghibli-style LoRA for Stable Diffusion v1.5.
# Usage: .\train_ghibli_lora.ps1

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

& "d:\FinetuningAI\venv\Scripts\accelerate.exe" launch "d:\FinetuningAI\train_text_to_image_lora.py" `
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" `
  --dataset_name="uwunish/ghibli-dataset" `
  --caption_column="text" `
  --image_column="image" `
  --resolution=512 --center_crop --random_flip `
  --train_batch_size=1 `
  --gradient_accumulation_steps=4 `
  --gradient_checkpointing `
  --mixed_precision="fp16" `
  --learning_rate=1e-04 `
  --lr_scheduler="cosine" --lr_warmup_steps=0 `
  --max_train_steps=1500 `
  --rank=16 `
  --seed=42 `
  --dataloader_num_workers=0 `
  --checkpointing_steps=500 `
  --validation_prompt="a studio ghibli style image of a girl standing in a green meadow, blue sky" `
  --num_validation_images=2 `
  --validation_epochs=50 `
  --output_dir="d:\FinetuningAI\ghibli-lora"
