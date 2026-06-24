# Stable Diffusion v1.5 — Studio Ghibli Style (LoRA Fine-Tuning)

Fine-tuning model **Stable Diffusion v1.5** menggunakan metode **LoRA (Low-Rank
Adaptation)** agar mampu menghasilkan gambar dengan gaya visual khas **Studio
Ghibli**.

Proyek ini mencakup script lengkap untuk **training**, **inference**, dan
**evaluasi** (perbandingan model dasar vs. model hasil fine-tuning).

## Daftar Isi
- [Fitur](#fitur)
- [Hasil](#hasil)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
  - [1. Training](#1-training)
  - [2. Generate Gambar (Inference)](#2-generate-gambar-inference)
  - [3. Evaluasi / Perbandingan](#3-evaluasi--perbandingan)
- [Dataset](#dataset)
- [Konfigurasi Training](#konfigurasi-training)
- [Cara Kerja](#cara-kerja)
- [Struktur Proyek](#struktur-proyek)
- [Catatan](#catatan)
- [Kredit](#kredit)

## Fitur
- Fine-tuning SD1.5 dengan LoRA — ringan dan hemat memori (cukup GPU 8GB VRAM).
- Script training siap pakai berbasis library `diffusers`.
- Script inference dengan peningkatan kualitas (VAE khusus + sampler DPM++ 2M Karras).
- Script evaluasi untuk membandingkan model dasar vs. hasil fine-tuning secara adil.

## Hasil
Perbandingan output dengan prompt, seed, dan parameter yang **identik** — satu-satunya
perbedaan adalah penggunaan LoRA hasil fine-tuning:

| Model Dasar (SD1.5) | SD1.5 + LoRA Ghibli |
|:---:|:---:|
| Gaya fotorealistis | Gaya lukisan tangan khas Ghibli |

Lihat file `comparison*.png` untuk contoh hasil.

## Prasyarat
- Python 3.10+
- GPU NVIDIA dengan CUDA (disarankan VRAM ≥ 8GB)
- Driver NVIDIA yang mendukung CUDA 12.4

## Instalasi

**1. Clone / unduh repository ini, lalu masuk ke foldernya**
```bash
cd FinetuningAI
```

**2. Buat dan aktifkan virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

**3. Install PyTorch (versi CUDA 12.4)**
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

**4. Install dependensi lainnya**
```bash
pip install -r requirements.txt
```

**5. (Opsional) Verifikasi GPU terdeteksi**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Output `True` berarti GPU siap digunakan.

## Cara Penggunaan

### 1. Training
Melatih LoRA dari awal menggunakan dataset Studio Ghibli.

```bash
# Windows (PowerShell)
.\train_ghibli_lora.ps1
```
- Saat pertama dijalankan, model dasar dan dataset akan diunduh otomatis.
- Hasil training disimpan di `ghibli-lora/pytorch_lora_weights.safetensors`.
- Durasi referensi: ~34 menit pada GPU RTX 4060 (8GB).

> Untuk uji coba cepat, ubah `--max_train_steps=1500` menjadi nilai lebih kecil
> (mis. `300`) di dalam `train_ghibli_lora.ps1`.

### 2. Generate Gambar (Inference)
Menghasilkan gambar menggunakan model hasil fine-tuning.

```bash
python run_ghibli.py
```
- Edit variabel `PROMPT` di dalam file untuk mengganti deskripsi gambar.
- Awali prompt dengan `"a studio ghibli style image of ..."` agar gaya Ghibli muncul kuat.
- Atur `LORA_SCALE` (0.0–1.2) untuk mengontrol kekuatan gaya.
- Hasil disimpan sebagai `ghibli_output.png`.

Untuk membandingkan, model dasar tanpa LoRA dapat dijalankan dengan:
```bash
python run_sd15.py
```

### 3. Evaluasi / Perbandingan
Menghasilkan gambar berdampingan: model dasar vs. model + LoRA, dengan parameter identik.

```bash
python compare_ghibli.py
```
- Output: `comparison.png` (gabungan), `compare_base.png`, dan `compare_lora.png`.

## Dataset
- **Sumber:** [`uwunish/ghibli-dataset`](https://huggingface.co/datasets/uwunish/ghibli-dataset)
- **Jumlah:** 913 gambar (512×512)
- **Format:** kolom `image` + `text` (caption)
- Caption diawali frasa *"a studio ghibli style image of ..."*, yang berfungsi
  sebagai **trigger phrase** saat inference.

## Konfigurasi Training
| Parameter | Nilai | Keterangan |
|---|---|---|
| Resolusi | 512×512 | Resolusi native SD1.5 |
| Batch size | 1 | Hemat VRAM |
| Gradient accumulation | 4 | Efektif batch = 4 |
| Gradient checkpointing | Aktif | Hemat VRAM |
| Mixed precision | fp16 | Hemat VRAM & lebih cepat |
| Learning rate | 1e-4 | — |
| LR scheduler | cosine | — |
| LoRA rank | 16 | Kapasitas adapter |
| Max train steps | 1500 | ≈ 6.5 epoch |

## Cara Kerja

### Konsep Stable Diffusion
Stable Diffusion menghasilkan gambar melalui proses **denoising bertahap**:
dimulai dari noise acak, model menghapus noise sedikit demi sedikit pada setiap
langkah hingga terbentuk gambar yang utuh. Proses ini dipandu oleh teks prompt
dan berlangsung di *latent space* (representasi gambar yang terkompresi) agar
hemat memori. Tiga komponen utamanya:

| Komponen | Tugas |
|---|---|
| Text Encoder (CLIP) | Mengubah prompt teks menjadi representasi numerik |
| UNet | Memprediksi dan menghapus noise pada setiap langkah |
| VAE | Mengubah hasil latent menjadi gambar pixel |

### Peran LoRA
Pada full fine-tuning, seluruh bobot UNet dilatih ulang sehingga membutuhkan
memori besar. **LoRA** membekukan bobot asli UNet dan hanya melatih matriks
adapter kecil yang ditambahkan ke model. Hanya adapter inilah yang mempelajari
gaya Studio Ghibli, sehingga ukuran file hasil kecil (~12 MB) dan proses training
muat di GPU 8GB. Saat inference, adapter ini digabungkan kembali ke UNet asli.

### Alur Training
Pada setiap langkah (`train_text_to_image_lora.py`):
1. Sebuah gambar beserta caption-nya diambil dari dataset.
2. Gambar di-*encode* menjadi latent, lalu ditambahkan noise acak.
3. UNet (beserta adapter LoRA) memprediksi noise tersebut dengan panduan caption.
4. Selisih antara prediksi dan noise asli dihitung sebagai *loss*.
5. Hanya bobot adapter LoRA yang diperbarui melalui backpropagation.
6. Langkah diulang hingga `max_train_steps` tercapai.

### Alur Inference
Pada `run_ghibli.py`:
1. Pipeline (text encoder + UNet + VAE) dimuat, lalu VAE dan sampler diganti
   dengan versi yang lebih baik.
2. Bobot LoRA dimuat dan digabungkan ke UNet (`load_lora_weights`).
3. Dari noise awal (ditentukan oleh `seed`), model melakukan denoising sebanyak
   `num_inference_steps`, dipandu oleh prompt dan negative prompt.
4. Hasil latent di-*decode* oleh VAE menjadi gambar dan disimpan.

Parameter inference penting:
| Parameter | Fungsi |
|---|---|
| `num_inference_steps` | Jumlah langkah denoising (lebih banyak = lebih detail, lebih lambat) |
| `guidance_scale` | Tingkat kepatuhan terhadap prompt |
| `LORA_SCALE` | Kekuatan gaya LoRA yang diterapkan |
| `seed` | Menentukan noise awal (untuk reproduktibilitas) |

## Struktur Proyek
```
FinetuningAI/
├── README.md                    # Dokumentasi
├── requirements.txt             # Daftar dependensi
├── .gitignore
├── train_ghibli_lora.ps1        # Launcher training
├── train_text_to_image_lora.py  # Script training LoRA (diffusers)
├── run_sd15.py                  # Inference model dasar
├── run_ghibli.py                # Inference model + LoRA
├── compare_ghibli.py            # Evaluasi base vs fine-tuned
├── ghibli-lora/
│   └── pytorch_lora_weights.safetensors   # Bobot LoRA hasil training
└── comparison*.png              # Hasil perbandingan base vs fine-tuned
```

## Catatan
- Model dasar yang digunakan: `stable-diffusion-v1-5/stable-diffusion-v1-5`.
- Stable Diffusion v1.5 hanya membaca 77 token pertama dari prompt; gunakan
  prompt yang ringkas.

## Kredit
- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- Dataset: [`uwunish/ghibli-dataset`](https://huggingface.co/datasets/uwunish/ghibli-dataset)
