# 🇮🇩 AksaraLLM — Panduan Training Komunitas

> **Model bahasa Indonesia pertama yang dibangun bersama komunitas!**
> Pipeline lengkap: Tokenizer → Pre-training → SFT → DPO — semuanya GRATIS di Google Colab.

---

## 📋 Overview Sistem

```mermaid
graph LR
    A[🔤 Tokenizer] --> B[📚 Pre-training]
    B --> C[🎓 SFT]
    C --> D[🎯 DPO]
    D --> E[🚀 Upload HF]
    
    style A fill:#4CAF50,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#9C27B0,color:#fff
    style E fill:#F44336,color:#fff
```

| Tahap | Waktu | Input | Output |
|-------|-------|-------|--------|
| 🔤 Tokenizer | 10 menit | Wikipedia ID | `vocab.json` + `merges.txt` |
| 📚 Pre-training | 60 menit | Wiki tokens | `pretrain_best.pt` |
| 🎓 SFT | 20 menit | Alpaca ID 52k | `sft_best.pt` |
| 🎯 DPO | 20 menit | hh-rlhf 160k | `dpo_best.pt` |
| **Total** | **~2 jam** | | **Model siap pakai!** |

---

## 🏗️ Arsitektur Model

```
AksaraLLM Kiel-Mini (26.4M params)
├── Vocab: 32,000 (BPE Indonesia)
├── Layers: 6
├── Heads: 6  
├── Embedding: 384
├── MLP Inner: 1,536
├── Max Seq Len: 256
└── Dropout: 0.05
```

---

## 🤝 Cara Komunitas Berkontribusi

### Model 1: Distributed Pre-training (Parallel Accounts)

Setiap anggota komunitas bisa training **shard berbeda** dari dataset secara paralel:

```
Anggota A (Akun 1): Wikipedia artikel 0-200k
Anggota B (Akun 2): Wikipedia artikel 200k-400k  
Anggota C (Akun 3): Wikipedia artikel 400k-665k
```

Hasilnya di-merge menjadi satu model.

### Model 2: Pipeline Relay

Setiap anggota melanjutkan training dari checkpoint sebelumnya:

```
Anggota A: Pre-training epoch 1 → upload checkpoint
Anggota B: Pre-training epoch 2-3 → upload checkpoint  
Anggota C: SFT → upload checkpoint
Anggota D: DPO → upload final model
```

### Model 3: Data Contribution

Anggota yang tidak punya GPU bisa bantu:
- 📝 Menulis Q&A bahasa Indonesia berkualitas tinggi
- 🔍 Review/validasi output model
- 📊 Benchmark dan evaluasi
- 📖 Dokumentasi dan tutorial

---

## 📦 Yang Dibutuhkan Setiap Kontributor

### Akun & Tools
- [x] Google Account (untuk Colab gratis)
- [x] GitHub Account (untuk clone repo)
- [x] HuggingFace Account (untuk upload model)

### File di Google Drive
```
MyDrive/
└── aksaraLLM-data/
    ├── wiki_tokens.npy              (dari koordinator)
    ├── hh_rlhf_COMPLETE_160k.jsonl  (dari koordinator)
    └── aksara-tokenizer-id/
        ├── vocab.json
        └── merges.txt
```

> [!IMPORTANT]
> Koordinator (admin) harus share folder `aksaraLLM-data` ke semua anggota via Google Drive sharing.

---

## 🚀 Step-by-Step: Full Pipeline

### STEP 0: Setup (Semua Anggota)

```python
# Cell 1 — Jalankan di Colab baru
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/AksaraLLM/aksaraLLM.git /content/aksaraLLM
!pip install datasets tokenizers transformers -q

import torch
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")
```

---

### STEP 1: Train Tokenizer (Sekali Saja — oleh Koordinator)

```python
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import os

print("📥 Download Wikipedia Indonesia...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.id", split="train")
print(f"✅ {len(wiki):,} artikel")

# Simpan ke file
with open("/tmp/wiki_id.txt", "w") as f:
    for art in wiki:
        if len(art["text"]) > 100:  # filter pendek
            f.write(art["text"] + "\n")

print("🔤 Training tokenizer...")
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["/tmp/wiki_id.txt"],
    vocab_size=32_000,
    min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>",
                    "### Instruksi:", "### Jawaban:", "### Respons:"]
)

tok_dir = "/content/drive/MyDrive/aksaraLLM-data/aksara-tokenizer-id"
os.makedirs(tok_dir, exist_ok=True)
tokenizer.save_model(tok_dir)
print(f"✅ Tokenizer tersimpan! Vocab: {tokenizer.get_vocab_size():,}")
```

---

### STEP 2: Pre-tokenize Wikipedia (Sekali Saja)

```python
import numpy as np
from tokenizers import ByteLevelBPETokenizer

tok_dir = "/content/drive/MyDrive/aksaraLLM-data/aksara-tokenizer-id"
tok = ByteLevelBPETokenizer(f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt")

SEQ_LEN = 256
all_ids = []
for i, art in enumerate(wiki):
    ids = tok.encode(art["text"]).ids
    if len(ids) > 50:
        all_ids.extend(ids)
    if (i+1) % 50000 == 0:
        print(f"  {i+1:,} articles processed...")

total = (len(all_ids) // SEQ_LEN) * SEQ_LEN
data = np.array(all_ids[:total], dtype=np.uint16).reshape(-1, SEQ_LEN)
np.save("/content/drive/MyDrive/aksaraLLM-data/wiki_tokens.npy", data)
print(f"✅ {data.shape[0]:,} sequences | {total/1e6:.0f}M tokens")
```

---

### STEP 3: Turbo Pre-training 🔥

```python
import numpy as np, time, math, os, torch
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.insert(0, "/content/aksaraLLM")
from aksarallm.model import aksaraLLMModel
from aksarallm.config import aksaraLLMConfig

device = "cuda"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Load data
data = np.load("/content/drive/MyDrive/aksaraLLM-data/wiki_tokens.npy").astype(np.int64)
print(f"✅ Data: {data.shape[0]:,} seqs | {data.shape[0]*256/1e6:.0f}M tokens")

# Model config
config = aksaraLLMConfig(
    vocab_size=32_000,
    n_layers=6, n_heads=6, n_embd=384, n_inner=1536,
    max_seq_len=256, dropout=0.05
)
model = aksaraLLMModel(config).to(device)
if hasattr(torch, "compile"):
    model = torch.compile(model, mode="reduce-overhead")
print(f"🔥 {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Hyperparameters
BATCH_SIZE, GRAD_ACCUM = 48, 3   # effective batch = 144
EPOCHS, LR_MAX, LR_MIN = 3, 6e-4, 6e-5

dataset = TensorDataset(torch.from_numpy(data))
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     pin_memory=True, num_workers=2,
                     persistent_workers=True, drop_last=True)

STEPS_PER_EPOCH = len(loader) // GRAD_ACCUM
TOTAL_STEPS     = STEPS_PER_EPOCH * EPOCHS
WARMUP_STEPS    = int(TOTAL_STEPS * 0.05)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX,
                               betas=(0.9, 0.95), weight_decay=0.1)

def get_lr(step):
    if step < WARMUP_STEPS: return LR_MAX * step / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * p))

scaler = torch.cuda.amp.GradScaler()
step = 0; t0 = time.time(); best_loss = 999
model.train()
print(f"🚀 Training: {TOTAL_STEPS:,} steps | ~60 menit")

for epoch in range(EPOCHS):
    epoch_loss = 0; count = 0
    for i, (batch,) in enumerate(loader):
        x, y = batch[:,:-1].to(device), batch[:,1:].to(device)
        with torch.cuda.amp.autocast():
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        epoch_loss += loss.item() * GRAD_ACCUM; count += 1

        if (i+1) % GRAD_ACCUM == 0:
            lr = get_lr(step)
            for pg in optimizer.param_groups: pg["lr"] = lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 200 == 0:
                elapsed = time.time()-t0
                avg = epoch_loss/count
                eta = (TOTAL_STEPS-step)/(step/elapsed)
                print(f"Ep{epoch+1} | {step}/{TOTAL_STEPS} | Loss:{avg:.3f} | LR:{lr:.1e} | ETA:{eta/60:.0f}m")

    avg = epoch_loss/count
    if avg < best_loss:
        best_loss = avg
        torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__},
                   "/content/drive/MyDrive/aksaraLLM-data/pretrain_best.pt")
        print(f"🏆 Epoch {epoch+1} | Best Loss: {best_loss:.3f}")

print(f"🎉 Pre-training selesai! | {(time.time()-t0)/60:.0f}m | Loss: {best_loss:.3f}")
```

---

### STEP 4: SFT (Supervised Fine-Tuning)

```python
import json, time, torch, sys, copy
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
sys.path.insert(0, "/content/aksaraLLM")
from aksarallm.model import aksaraLLMModel
from aksarallm.config import aksaraLLMConfig

tok_dir = "/content/drive/MyDrive/aksaraLLM-data/aksara-tokenizer-id"
tok = ByteLevelBPETokenizer(f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt")
device = "cuda"

# Download Alpaca Indonesia
alpaca = load_dataset("cahya/alpaca-id", split="train")
print(f"✅ Alpaca: {len(alpaca):,} samples")

# Identity data — model tahu siapa dirinya!
identity_texts = [
    "Di bawah ini adalah instruksi. Tulis tanggapan yang tepat.\n\n### Instruksi:\nSiapa kamu?\n\n### Respons:\nSaya adalah AksaraLLM, asisten AI berbahasa Indonesia.",
    "Di bawah ini adalah instruksi. Tulis tanggapan yang tepat.\n\n### Instruksi:\nApa itu AksaraLLM?\n\n### Respons:\nAksaraLLM adalah model bahasa Indonesia open-source.",
    "Di bawah ini adalah instruksi. Tulis tanggapan yang tepat.\n\n### Instruksi:\nSiapa yang membuat kamu?\n\n### Respons:\nSaya dibuat oleh komunitas AksaraLLM.",
] * 80  # repeat agar hafal

class SFTDataset(Dataset):
    def __init__(self, hf_data, identities, tokenizer, max_len=256):
        self.samples = []
        texts = [s["text"] for s in hf_data] + identities
        for t in texts:
            ids = tokenizer.encode(t).ids[:max_len]
            if len(ids) > 5:
                self.samples.append(torch.tensor(ids, dtype=torch.long))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate(batch):
    ml = max(x.size(0) for x in batch)
    pad = torch.zeros(len(batch), ml, dtype=torch.long)
    for i, x in enumerate(batch): pad[i,:x.size(0)] = x
    return pad

dataset = SFTDataset(alpaca, identity_texts, tok)
loader  = DataLoader(dataset, batch_size=32, shuffle=True,
                     collate_fn=collate, num_workers=2, pin_memory=True)

# Load pretrained
ckpt = torch.load("/content/drive/MyDrive/aksaraLLM-data/pretrain_best.pt",
                  map_location=device, weights_only=False)
cfg = aksaraLLMConfig(**{k:v for k,v in ckpt["config"].items()
                         if k in aksaraLLMConfig.__dataclass_fields__})
model = aksaraLLMModel(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
if hasattr(torch, "compile"):
    model = torch.compile(model, mode="reduce-overhead")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()
t0 = time.time(); best = 999

for epoch in range(2):
    model.train(); total = 0
    for step, batch in enumerate(loader):
        x, y = batch[:,:-1].to(device), batch[:,1:].to(device)
        with torch.cuda.amp.autocast():
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad()
        total += loss.item()
        if (step+1) % 200 == 0:
            print(f"Ep{epoch+1} | {step+1}/{len(loader)} | Loss:{total/(step+1):.4f}")
    avg = total/len(loader)
    if avg < best:
        best = avg
        torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__},
                   "/content/drive/MyDrive/aksaraLLM-data/sft_best.pt")
        print(f"🏆 SFT Epoch {epoch+1} | Loss: {best:.4f}")

print(f"🎉 SFT selesai! | {(time.time()-t0)/60:.0f}m")
```

---

### STEP 5: DPO (Direct Preference Optimization)

```python
import json, copy, time, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# (model, tokenizer, config sudah loaded dari SFT)

# Load SFT sebagai policy + reference
ckpt = torch.load("/content/drive/MyDrive/aksaraLLM-data/sft_best.pt",
                  map_location=device, weights_only=False)
cfg = aksaraLLMConfig(**{k:v for k,v in ckpt["config"].items()
                         if k in aksaraLLMConfig.__dataclass_fields__})
policy = aksaraLLMModel(cfg).to(device)
policy.load_state_dict(ckpt["model_state_dict"], strict=False)
ref = copy.deepcopy(policy)
for p in ref.parameters(): p.requires_grad_(False)
ref.eval()
if hasattr(torch, "compile"):
    policy = torch.compile(policy, mode="reduce-overhead")

class DPODataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.pairs = []
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    c = tokenizer.encode(d["chosen"]).ids[:max_len]
                    r = tokenizer.encode(d["rejected"]).ids[:max_len]
                    if len(c)>5 and len(r)>5 and c!=r:
                        self.pairs.append((c, r))
                except: pass
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

def pad(seqs):
    ml = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), ml, dtype=torch.long)
    for i,s in enumerate(seqs): out[i,:len(s)] = torch.tensor(s)
    return out

def collate_dpo(batch):
    ch, rj = zip(*batch)
    return pad(ch), pad(rj)

def log_probs(m, ids):
    x, y = ids[:,:-1], ids[:,1:]
    with torch.cuda.amp.autocast():
        logits, _ = m(x)
    return F.log_softmax(logits, -1).gather(2, y.unsqueeze(2)).squeeze(2).sum(1)

dpo_data = DPODataset("/content/drive/MyDrive/aksaraLLM-data/hh_rlhf_COMPLETE_160k.jsonl", tok)
dpo_loader = DataLoader(dpo_data, batch_size=16, shuffle=True,
                        collate_fn=collate_dpo, num_workers=2, pin_memory=True)

BETA = 0.1
opt  = torch.optim.AdamW(policy.parameters(), lr=5e-7, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()
best_acc = 0; t0 = time.time()

for step, (chosen, rejected) in enumerate(dpo_loader):
    chosen, rejected = chosen.to(device), rejected.to(device)
    policy.train()
    with torch.no_grad():
        ref_ch, ref_rj = log_probs(ref, chosen), log_probs(ref, rejected)
    pol_ch, pol_rj = log_probs(policy, chosen), log_probs(policy, rejected)
    ratio = BETA * ((pol_ch-ref_ch) - (pol_rj-ref_rj))
    loss  = -F.logsigmoid(ratio).mean()
    acc   = (ratio>0).float().mean().item()
    
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad()
    
    if (step+1)%200==0:
        print(f"DPO {step+1}/{len(dpo_loader)} | Loss:{loss.item():.4f} | Acc:{acc*100:.1f}%")
    if (step+1)%500==0 and acc>best_acc:
        best_acc = acc
        torch.save({"model_state_dict": policy.state_dict(), "config": cfg.__dict__},
                   "/content/drive/MyDrive/aksaraLLM-data/dpo_best.pt")

print(f"🎉 DPO selesai! | {(time.time()-t0)/60:.0f}m | Best Acc: {best_acc*100:.1f}%")
```

---

### STEP 6: Test Model

```python
model.eval()
def chat(prompt, max_new=80, temp=0.7):
    text = f"Di bawah ini adalah instruksi.\n\n### Instruksi:\n{prompt}\n\n### Respons:\n"
    ids = tok.encode(text).ids[-100:]
    ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(ids[:, -256:])
            nxt = torch.multinomial(torch.softmax(logits[0,-1,:]/temp, -1), 1)
            ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
    return tok.decode(ids[0].tolist()).split("### Respons:")[-1].strip()

for q in ["Siapa kamu?", "Apa ibu kota Indonesia?", "Jelaskan fotosintesis"]:
    print(f"Q: {q}\nA: {chat(q)}\n{'─'*50}")
```

---

### STEP 7: Upload ke HuggingFace

```python
from huggingface_hub import HfApi, create_repo

HF_TOKEN = "hf_xxxx"  # Ganti dengan token kamu
api = HfApi()

create_repo("AksaraLLM/Kiel-Mini-26M-v3", token=HF_TOKEN,
            exist_ok=True, private=False)

# Upload model + tokenizer
for f, name in [
    ("/content/drive/MyDrive/aksaraLLM-data/dpo_best.pt", "dpo_best.pt"),
    (f"{tok_dir}/vocab.json", "tokenizer/vocab.json"),
    (f"{tok_dir}/merges.txt", "tokenizer/merges.txt"),
]:
    api.upload_file(path_or_fileobj=f, path_in_repo=name,
                    repo_id="AksaraLLM/Kiel-Mini-26M-v3", token=HF_TOKEN)

print("🎉 Model LIVE di HuggingFace!")
```

---

## 🎯 Tips untuk Kontributor

### Hemat Quota Colab

| Tip | Efek |
|-----|------|
| Backup `.npy` ke Drive | Hemat 15 menit re-tokenize |
| Pakai `torch.compile` | 20-40% lebih cepat |
| `drop_last=True` di DataLoader | Avoid OOM di batch terakhir |
| Monitor GPU RAM | Jangan melebihi 13GB |
| Restart session setelah OOM | Bersihkan memory leak |

### Cara Kontribusi Data

1. **Buat Q&A berkualitas** dalam format:
```json
{"instruction": "Apa itu demokrasi?", "output": "Demokrasi adalah sistem pemerintahan..."}
```

2. **Submit via Google Form** (link dari koordinator)
3. **Review PR di GitHub** — cek kualitas terjemahan

### Reporting Hasil

Setelah training selesai, share di Discord:
```
📊 Training Report
- Model: Kiel-Mini-26M-v3
- Pre-train loss: X.XXX
- SFT loss: X.XXXX  
- DPO accuracy: XX.X%
- Waktu total: XX menit
- GPU: T4 / L4 / A100
```

---

## 🔗 Links

- **GitHub**: [github.com/AksaraLLM](https://github.com/AksaraLLM)
- **HuggingFace**: [huggingface.co/AksaraLLM](https://huggingface.co/AksaraLLM)
- **Discord**: [Invite Link]

---

> [!NOTE]
> Dokumen ini akan terus diperbarui seiring perkembangan model.
> Kontribusi dokumentasi juga sangat diapresiasi! 🙏
