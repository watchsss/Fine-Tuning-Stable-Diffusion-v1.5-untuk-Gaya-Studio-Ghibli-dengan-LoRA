# Prompt untuk Claude Design — Pembuatan PPT

> **Cara pakai:** copy seluruh isi blok di bawah ke Claude Design, lalu lampirkan
> 10 gambar `comparison_1.png` … `comparison_10.png`.
>
> **Catatan:** pemetaan `comparison_N.png` ke prompt diasumsikan **berurutan**
> (comparison_1 = prompt #1, dst). Verifikasi dulu urutan file Anda; tukar teks
> prompt bila tidak sesuai.

---

Buatkan saya presentasi (PPT) profesional tentang proyek machine learning saya.
Bahasa: Indonesia. Audiens: dosen & mahasiswa (technical). Gaya: clean, modern,
minimalis, tema warna terinspirasi Studio Ghibli (hijau pastel, biru langit, krem).
Sertakan judul slide yang jelas dan poin-poin ringkas (bukan paragraf panjang).

JUDUL PROYEK: "Fine-Tuning Stable Diffusion v1.5 untuk Gaya Studio Ghibli dengan LoRA"

Saya melampirkan 10 gambar perbandingan (comparison_1.png s/d comparison_10.png).
Setiap gambar menunjukkan hasil model DASAR (kiri) vs model HASIL FINE-TUNING (kanan)
dengan prompt yang sama. Gunakan gambar-gambar ini di slide hasil sesuai pemetaan di bawah.

Buat slide berikut:

--- SLIDE 1: Judul ---
- Judul proyek
- Subjudul: "Fine-tuning dengan metode LoRA pada GPU 8GB"
- Nama saya / NIM (kosongkan placeholder)

--- SLIDE 2: Latar Belakang & Tujuan ---
- Tujuan: mengajarkan gaya visual Studio Ghibli ke model Stable Diffusion v1.5
- Tantangan: full fine-tuning butuh VRAM ~24GB, sedangkan GPU hanya 8GB (RTX 4060)
- Solusi: metode LoRA (Low-Rank Adaptation)

--- SLIDE 3: Cara Kerja Stable Diffusion ---
- Proses denoising bertahap: dari noise acak -> gambar utuh, dipandu teks
- Bekerja di latent space agar hemat memori
- 3 komponen: Text Encoder (CLIP), UNet (denoiser), VAE (decoder ke pixel)

--- SLIDE 4: Apa itu LoRA & Kenapa Dipakai ---
- Full fine-tuning melatih SELURUH bobot UNet -> berat
- LoRA membekukan bobot asli, hanya melatih adapter kecil tambahan
- Keuntungan: muat di 8GB VRAM, training ~34 menit, file hasil hanya ~12 MB

--- SLIDE 5: Dataset ---
- Sumber: uwunish/ghibli-dataset (Hugging Face Hub)
- 913 gambar, resolusi 512x512, dengan caption
- Trigger phrase: "a studio ghibli style image of ..."

--- SLIDE 6: Konfigurasi Training ---
Tampilkan sebagai tabel:
| Parameter | Nilai |
| Resolusi | 512x512 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Mixed precision | fp16 |
| Learning rate | 1e-4 |
| LR scheduler | cosine |
| LoRA rank | 16 |
| Max train steps | 1500 (~6.5 epoch) |

--- SLIDE 7: Kode Training (launcher) ---
Tampilkan snippet kode ini (gaya code block):

accelerate launch train_text_to_image_lora.py `
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" `
  --dataset_name="uwunish/ghibli-dataset" `
  --caption_column="text" --resolution=512 --random_flip `
  --train_batch_size=1 --gradient_accumulation_steps=4 `
  --gradient_checkpointing --mixed_precision="fp16" `
  --learning_rate=1e-04 --lr_scheduler="cosine" `
  --max_train_steps=1500 --rank=16 --seed=42 `
  --output_dir="ghibli-lora"

--- SLIDE 8: Alur Training (proses belajar) ---
- Ambil gambar + caption dari dataset
- Encode ke latent, tambahkan noise acak
- UNet + adapter LoRA menebak noise tersebut (dipandu caption)
- Hitung loss = selisih tebakan vs noise asli
- Update HANYA bobot adapter LoRA (backpropagation)
- Ulangi hingga 1500 step

--- SLIDE 9: Kode Inference ---
Tampilkan snippet kode ini (gaya code block):

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", vae=vae, torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)
pipe.load_lora_weights("ghibli-lora")   # <- memasang hasil fine-tuning

image = pipe(prompt, negative_prompt=neg, num_inference_steps=25,
             guidance_scale=7.5, cross_attention_kwargs={"scale": 0.9}).images[0]

--- SLIDE 10: Metode Evaluasi ---
- Bandingkan model dasar vs hasil fine-tuning
- Prompt, seed, sampler, VAE, step DIBUAT IDENTIK
- Satu-satunya variabel yang berbeda: penggunaan LoRA
- Tujuan: memastikan perubahan gaya MURNI dari fine-tuning (apple-to-apple)

--- SLIDE 11-15: Hasil Perbandingan (Galeri) ---
Setiap slide menampilkan 2 gambar comparison (kiri = base SD1.5, kanan = + LoRA Ghibli)
beserta prompt-nya. Tampilkan gambar besar dan prompt sebagai caption di bawahnya.
Pemetaan gambar -> prompt:

- comparison_1.png  | "a studio ghibli style image of a girl with a red umbrella walking on a countryside path, rice fields, blue sky with big clouds"
- comparison_2.png  | "a studio ghibli style image of a cozy cottage on a green hill, flower meadow, windmill, soft morning light, fluffy clouds"
- comparison_3.png  | "a studio ghibli style image of a small seaside town, boats on calm water, red rooftops, seagulls, warm sunset sky"
- comparison_4.png  | "a studio ghibli style image of a magical forest with giant trees, glowing spirits, mossy rocks, sunbeams through leaves"
- comparison_5.png  | "a studio ghibli style image of an old train crossing green countryside, mountains in background, bright summer day"
- comparison_6.png  | "a studio ghibli style image of a boy sitting on a grassy hill looking at the sky, kites flying, vast clouds, peaceful afternoon"
- comparison_7.png  | "a studio ghibli style image of a warm cozy kitchen, steaming food on the table, sunlight from window, detailed interior"
- comparison_8.png  | "a studio ghibli style image of a rainy city street at night, glowing lanterns, reflections on wet road, a girl with umbrella"
- comparison_9.png  | "a studio ghibli style image of a wide flower field swaying in the wind, a single tree, blue sky, cinematic wide shot"
- comparison_10.png | "a studio ghibli style image of a floating castle above the clouds, airships in the distance, dramatic sky, fantasy landscape"

Atur 2 gambar per slide agar muat (total ~5 slide galeri).

--- SLIDE 16: Kesimpulan ---
- Fine-tuning LoRA berhasil mentransfer gaya Studio Ghibli ke SD1.5
- Efisien: cukup GPU 8GB, ~34 menit, file 12 MB
- Hasil perbandingan menunjukkan perubahan gaya yang konsisten & jelas
- Pengembangan lanjut: naikkan rank/steps, coba checkpoint komunitas

--- SLIDE 17: Tech Stack ---
- Python, PyTorch (CUDA), Hugging Face Diffusers, PEFT (LoRA), Accelerate
- Model dasar: Stable Diffusion v1.5
- GPU: NVIDIA RTX 4060 (8GB VRAM)

Gunakan ikon/ilustrasi sederhana bila relevan, dan jaga konsistensi warna di semua slide.
