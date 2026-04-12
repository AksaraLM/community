# GPU Donation Guide

Thank you for considering donating compute resources to aksaraLLM! 🙏

GPU compute is the single most critical resource for training large language models. Your contribution directly enables open-source AI development.

## How to Donate

### Option 1: Cloud Credits
If you have unused cloud credits (AWS, GCP, Azure, Lambda Labs, CoreWeave, etc.), you can donate them to our training runs.

**Process:**
1. Contact us on Discord (`#gpu-donations` channel)
2. We'll coordinate the setup
3. You'll get credited in our acknowledgments

### Option 2: Direct Hardware Access
If you have GPUs that are idle (university clusters, personal rigs, company servers), you can contribute them to our distributed training pool.

**Minimum Requirements:**
- NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A100, H100)
- Stable internet connection (100Mbps+ recommended)
- Ability to run Docker containers
- Available for sustained periods (>24 hours recommended)

**Process:**
1. Join Discord and go to `#gpu-donations`
2. Fill out the GPU Registration Form (link in Discord)
3. We'll provide setup instructions
4. Your node joins our training cluster

### Option 3: Organizational Sponsorship
If your organization can provide sustained compute access:

**What we need:**
- A100/H100 cluster access for training runs
- Estimated need: 64-256 GPUs for 2-4 weeks per major training run
- We're flexible on scheduling

**What you get:**
- Logo on our website and README
- Acknowledgment in all published papers/reports
- Early access to model checkpoints
- Co-authorship opportunity on technical reports

**Contact:** sponsors@aksarallm.org (or Discord DM to Project Lead)

## Compute Needs by Phase

| Phase | What | GPUs Needed | Duration |
|-------|------|-------------|----------|
| Phase 3 | Small experiments (125M-1B) | 1-8x A100 | 1-2 weeks |
| Phase 4 | Main training (7B) | 64-128x A100 | 2-4 weeks |
| Phase 5 | Fine-tuning (SFT/DPO) | 8-16x A100 | 1 week |
| Future | Scaling (13B-70B) | 256-512x A100 | 4-8 weeks |

## Grant Programs We're Applying To

We're actively applying to these compute grant programs:

- [ ] **Google TPU Research Cloud (TRC)** — Free TPU v4 pods
- [ ] **Microsoft Accelerate Foundation Models Research**
- [ ] **NVIDIA Academic GPU Grant Program**
- [ ] **Meta Research Credits**
- [ ] **Amazon Research Credits**
- [ ] **Lambda Labs Open Source Program**
- [ ] **CoreWeave Open Source Sponsorship**
- [ ] **HuggingFace Compute Sponsorship**
- [ ] **University HPC partnerships**

## Security & Trust

- All donated compute is used **exclusively** for aksaraLLM training
- We publish detailed training logs showing exactly what ran
- Docker containers are open-source and auditable
- We never store any data beyond what's needed for training
- Donors can revoke access at any time

## Recognition

All compute donors are recognized in:
- 🏆 Our website's sponsors page
- 📝 Technical reports and papers
- 🎉 Release announcements
- 💜 Special role on Discord

### Current Sponsors & Donors
*Be the first! Join us on [Discord](https://discord.gg/aksarallm).*

---

## FAQ

**Q: Can I donate consumer GPUs (RTX 3090, 4090)?**
A: Yes! For small experiments and evaluation. For main training, we need data center GPUs, but consumer GPUs are still very valuable for experimentation.

**Q: How much would it cost to train aksaraLLM-7B?**
A: Approximately $50,000-$150,000 in cloud GPU costs, depending on efficiency and hardware. This is why donations and grants are so important.

**Q: What if I can only donate for a few hours?**  
A: Every bit helps! Even short donations are useful for evaluation runs and small experiments.

**Q: Is my data safe?**
A: We don't access any data on your machine. Training runs execute in isolated Docker containers. All code is open-source and auditable.

---

*Questions? Reach out on Discord `#gpu-donations` or email sponsors@aksarallm.org*
