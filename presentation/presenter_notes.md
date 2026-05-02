# TruPhoto — Presenter Notes

Companion notes for `truphoto_presentation.pptx`. Each section corresponds to one slide and includes:

- **Time**: target speaking duration
- **Goal**: what the audience should walk away knowing
- **Speaking points**: what to actually say (paraphrase, don't read aloud)
- **If asked**: anticipated Q&A

Total target runtime: **15–17 minutes** (15 slides), leaving 3–5 minutes for Q&A.

---

## Slide 1 — Title

**Time:** 30 seconds

**Speaking points:**
- "Hi, I'm Tabeen. This project is TruPhoto."
- "It's a 3-class image authenticity classifier — Real, Forged, AI-Generated — built for CS483."
- "Final result: **74.92% validation accuracy and 73.27% test accuracy** across 25 different AI generators on a frozen-feature pipeline that needs no GPU. The 1.65-point val/test gap is small — the model generalizes."
- "The numbers are deliberately about 3 points lower than an earlier version because we removed a dataset-source shortcut. I'll walk through how we got here, including four iterations and what each one taught us."

---

## Slide 2 — Why image authenticity matters

**Time:** 60 seconds

**Goal:** Establish that the *three-class* framing is non-trivial and motivated.

**Speaking points:**
- The synthetic-image problem now has two distinct shapes: classical forgery (splicing, copy-move) and full generative-AI synthesis.
- Most academic work treats this as binary — real vs. fake — which collapses two very different threat models into one.
- A real moderation pipeline cares about the difference: a spliced image came from a human editor; an AI-generated one came from a model. The investigative response is different.
- TruPhoto explicitly outputs three classes so the system gives a useful signal, not just "fake or not."

**If asked:** "Why not split AI by generator?"
> Generator counts grow over time and we'd need per-generator labels for everything. Three classes is the highest-resolution split where the labels are stable and the training data is available.

---

## Slide 3 — Pipeline overview

**Time:** 60 seconds

**Goal:** Establish the two-stage architecture before the audience sees the code-level steps.

**Speaking points:**
- Stage 1: a frozen pretrained CNN turns each image into a fixed-length feature vector.
- Stage 2: a classical classifier (Random Forest or SVM) maps that vector to a class.
- The CNN runs once per image; the classifier iterates fast, so we can try many variants without re-running ImageNet inference.
- Standard transfer learning. We use ImageNet's general visual knowledge for free, only train the small final classifier on our task.
- Trade-off: we lose the ability to fine-tune CNN weights on forgery-specific cues. We'll come back to this on the design-decisions slide and again on the explicit fine-tuning slide.

**If asked:** "Why not just a softmax head on the CNN?"
> A softmax head would require backprop through the CNN — that brings GPU dependency and longer training. Two-stage with a classical classifier was a deliberate choice for fast iteration on a CPU-only setup.

---

## Slide 4 — Datasets

**Time:** 75 seconds

**Goal:** Make the dataset boundaries explicit. This sets up the bias-fix story on the next slide.

**Speaking points:**
- Two Kaggle datasets: ArtiFact and CASIA 2.0.
- ArtiFact has roughly 33 generator subfolders. After our partial download, 25 contribute usable AI images. Several folders contribute to both classes via `target=0` for real reference images and a separate target index for the AI outputs.
- 10 ArtiFact folders contribute Real images (afhq, ffhq, coco, lsun, imagenet, celebahq, metfaces, landscape, pro_gan, cycle_gan).
- CASIA 2.0 contributes the **Forged** class via its `Tp/` (tampered) folder — 5,123 images, the full set.
- **CASIA 2.0 also contributes to the Real class via its `Au/` (authentic) folder — 7,491 images.** This was added in the final iteration to break a dataset-source shortcut.
- We sample 12,500 images per class. Forged is undersized (only 5,123 available). Stratified 70/15/15 split with seed 42.

**If asked:** "Why did the budget jump from 5K to 12.5K?"
> To test whether more training data per generator would lift the harder cases. It didn't — and that result becomes evidence on the design-decisions slide.

**If asked:** "Why use Au only now and not from the start?"
> The first three runs had Real entirely from ArtiFact and Forged entirely from CASIA. The cross-source check showed the model could short-circuit by predicting "Forged" whenever it saw a CASIA-style image. Mixing Au into Real forces it to look at actual content.

---

## Slide 5 — Step 1: Data preparation (with three key fixes)

**Time:** 90 seconds (this is the methodological-arc slide, slow down here)

**Goal:** Convey that the project found and fixed three non-obvious bugs in dataset handling — the difference between a "we built a model" story and a "we discovered something" story.

**Speaking points:**
- Three fixes, in the order they were found.
- **Fix #1 — generator-aware label inference.** ArtiFact's metadata uses two different schemas across its folders. Some use `target=0/1` for real/AI. Others use `target=<imagenet_class_index>` (big_gan uses target=6, latent_diffusion uses a different index, and so on). Our original discovery code only checked for binary targets — so for those folders, every row was silently skipped. Result: only `stylegan2` registered as an AI source on disk. After the fix (folder-name fallback for non-binary targets), discovery found 25 AI sources instead of 1.
- **Fix #2 — stratified sampling across sources.** Even with all 25 sources discovered, naive `random.sample` on the flat list of ~1.5M AI images allocated ~70% to stylegan2 alone (it was the largest source). The stratified sampler distributes equal quotas per source first, then redistributes leftovers from large pools.
- **Fix #3 — mix CASIA Au into the Real class.** This was the last fix. Earlier runs had Real entirely from ArtiFact and Forged entirely from CASIA. The model learned "this image came from CASIA" as a free predictor for Forged — visible when uploading landscape photos to the demo and getting "Forged" with high confidence. We added 7,491 CASIA authentic images into the Real class with a `casia2_` prefix for source tracking.
- Together, the three fixes turned the project from a stylegan2-detector with inflated accuracy into a real 25-generator classifier with honest accuracy. Slide 12 shows the iteration history with concrete numbers.

**If asked:** "How did you find the bugs?"
> Each one was caught by the cross-source sanity check (step4b). The first run flagged that AI was 100% stylegan2; the third run flagged that landscape photos were being routed to Forged. That's why I treat step4b as a methodological safety net, not just an evaluation.

---

## Slide 6 — Step 2: Feature extraction

**Time:** 60 seconds

**Goal:** Be very explicit that the CNN is frozen — this is THE single most-asked question.

**Speaking points:**
- We use `timm.create_model('xception', pretrained=True, num_classes=0)`.
- `num_classes=0` strips the final classification head; what's left ends at the global-average-pooling layer, giving a 2048-dim feature vector per image.
- Two important lines: `for p in model.parameters(): p.requires_grad = False` and `model.eval()`. **Nothing in the CNN is trained on our data.**
- Forward-pass with `torch.no_grad()`, save features as `.npy`. Train/val/test for X and y.
- For ablation we also extract MobileNetV2 features — chosen for size contrast, to test whether feature-extractor capacity actually matters here.
- On CPU, Xception extraction took ~70 minutes total; MobileNetV2 ~32 minutes total.

**If asked:** "Why not fine-tune?"
> That's slide 14 — there's a full slide on the explicit decision and three reasons.

---

## Slide 7 — Step 3 + 4: Classifiers and ablation

**Time:** 75 seconds

**Goal:** Walk through the ablation matrix and the actual numbers.

**Speaking points:**
- Two classifiers per backbone: Random Forest with 200 trees and balanced class weights; SVM with RBF kernel, C=10, scaled features.
- Test accuracy on the 12,500-per-class run with CASIA Au merged into Real:
  - MobileNetV2 + RF → 69.81% val / **69.40% test**
  - MobileNetV2 + SVM → **74.92% val / 73.27% test — best combination**
  - Xception (both heads) → not retrained for this iteration. Every prior run had MobileNetV2 ahead of Xception on both classifiers, so we skip the multi-hour Xception SVM retrain and quote the consistent pattern.
- The val/test gap on the best combo is only 1.65 pp — good generalization, no overfitting.
- Per-class F1 on the best combo: Real 0.68, Forged 0.88, AI_Generated 0.72. Forged is the easiest class because splicing artifacts are well-aligned with ImageNet features.
- Step 4 outputs per-class precision/recall/F1, confusion matrices per combination, an ablation chart, and an evaluation report.

**If asked:** "Why does MobileNetV2 win?"
> Two reasons. First, the feature space is 1280-dim instead of 2048-dim — less room to overfit on ~21K training samples. Second, MobileNetV2's inverted-residual blocks preserve more low-level texture information than Xception's deeper depthwise-separable blocks. Forgery cues live in low-level texture, so the less-abstracted backbone wins.

**If asked:** "Why did the number drop from 77.62% to 74.92%?"
> CASIA Au was merged into Real. The earlier 77.62% was inflated by the model using "this came from CASIA" as a predictor for Forged. We gave up about 3 points of accuracy in exchange for the model actually classifying by image content. Slide 12 explains.

---

## Slide 8 — Ablation chart (visual)

**Time:** 30 seconds

**Goal:** Reinforce the headline visually before moving into framing/positioning slides.

**Speaking points:**
- "Same numbers as the previous slide, but as a chart so you can see the gap."
- Two bars: MobileNetV2 + RF at 69.40%, MobileNetV2 + SVM at 73.27%. SVM wins by ~4 points on the same features — that's the cost of using a tree ensemble on dense GAP vectors instead of a max-margin classifier.
- The dashed red line is the Ali et al. baseline at 92.23%, shown for reference only — different task entirely (binary, single dataset, fine-tuned). The next slide explains why a direct comparison would be misleading.

**If asked:** "Why isn't there an Xception bar?"
> The Xception classifiers were trained on the previous dataset iteration (before CASIA Au was merged into Real). Rather than carry stale numbers in the report, we removed those models. Every prior run had MobileNetV2 ahead of Xception on this task, so we retain MobileNetV2 as the headline backbone.

---

## Slide 9 — Comparison with Ali et al. (2022)

**Time:** 90 seconds (key academic-positioning slide)

**Goal:** Position TruPhoto honestly against the published baseline.

**Speaking points:**
- Ali et al. (2022) reported 92.23% on CASIA 2.0. Two big differences:
  - Theirs is binary (Real vs. Forged) on one dataset. Ours is 3-class on two datasets combined, with 25 different AI generators.
  - Theirs fine-tunes Xception end-to-end. Ours uses a frozen Xception or MobileNetV2 + classical classifier.
- A direct number comparison would be misleading. Two ways to honestly contextualize:
  - **Gain over chance:** binary chance = 50%, theirs is +42 over chance. 3-class chance = 33%, ours is +42 over chance. Comparable lift.
  - **Same subtask:** when we restrict our model to the binary CASIA Tp/ subtask, we hit ~89% after the Au merge (down from ~96% before). The 89% is more honest because the model can no longer rely on dataset signature.
- The 22-point accuracy gap between 92% and 75% is what the 3-class formulation costs us, plus the 25-generator AI diversity, plus the dataset-shortcut removal.

**If asked:** "Why didn't you reproduce their setup directly?"
> Time and GPU access are the practical constraints; the project also asked for a richer 3-class output, which forces the formulation to differ.

**If asked:** "Then what's the value of citing them?"
> Reference point — they've established frozen-Xception-style features carry useful forgery signal at ~92% on CASIA. We're benchmarking the same backbone on a harder problem, with explicit per-source breakdown that their paper doesn't have.

---

## Slide 10 — Generator difficulty tiers (the headline finding)

**Time:** 90 seconds (this is your headline — slow down)

**Goal:** Show that the 75% number isn't the interesting result — the *per-generator pattern* is.

**Speaking points:**
- The cross-source check breaks AI accuracy down by generator. Three clean tiers emerge:
  - **Easy generators** (≥85%): StarGAN 100%, sfhq 96%, face_synthetics 96%, DDPM 94%, CIPS 91%, gansformer 89%, denoising_diffusion_gan 86%, BigGAN 85%, palette 85%. Mostly older or artifact-heavy generators.
  - **Medium generators** (60–85%): gau_gan 76%, StyleGAN1 69%, LAMA 69%, StyleGAN2 64%, taming_transformer 62%, projected_gan 62%, GLIDE 60%, MAT 60%.
  - **Hard generators** (<60%): **Stable Diffusion 59%**, diffusion_gan 57%, latent_diffusion 55%, vq_diffusion 52%, generative_inpainting 46%, **StyleGAN3 33%**.
- The pattern is unmistakable: **older / artifact-heavy generators are detected easily; modern / photorealistic generators (StyleGAN3, modern diffusion) approach chance.**
- The StyleGAN gradient inside our own results says it best: StyleGAN1 69%, StyleGAN2 64%, StyleGAN3 33%. Each release was specifically designed to remove the artifacts of the previous one.
- This isn't a flaw in our pipeline — it's a finding about what frozen ImageNet features can and cannot do. Newer generators are explicitly designed to look photographic, and ImageNet's semantic features don't differentiate them.

**If asked:** "Why is StyleGAN3 worse than StyleGAN1?"
> StyleGAN3 added equivariance properties that eliminate the texture-sticking artifacts we'd otherwise detect. So our model's failure on StyleGAN3 is, in a sense, a measurement of how successful StyleGAN3's authors were.

---

## Slide 11 — Key design decisions

**Time:** 60 seconds

**Goal:** Show that every choice was deliberate and tested.

**Speaking points:**
- Frozen vs. fine-tuned: trades accuracy for speed and reproducibility. CPU-friendly. Misses forgery-specific cues — slide 14 has the explicit decision.
- RF + SVM vs. neural head: classical classifiers are interpretable and fast; we lose neural-net flexibility but gain easy calibration.
- 3-class vs. binary: closer to a real product use case, harder to compare to prior work.
- The data-scaling experiment is the one I want to highlight: when we went from 5,000 to 12,500 samples per class, **Random Forest accuracy dropped on both backbones** — Xception −2.4 pp, MobileNetV2 −2.7 pp. SVM dropped similarly. That's direct empirical evidence the bottleneck is the frozen features, not the data. More data won't fix this.
- Engineering: project is structured as a real Python package — `src/__init__.py`, `pyproject.toml`, absolute imports, model meta-manifests with git SHA for reproducibility.

---

## Slide 12 — Iteration history (what each run taught us)

**Time:** 90 seconds (slow down — this is the credibility slide)

**Goal:** Show that the headline number is the result of four deliberate iterations, each fixing a real methodological flaw.

**Speaking points:**
- Four runs, each one moved both the headline number AND fixed something real.
- **Run 1 — Naive baseline.** ~84% accuracy looked great. Reality: only stylegan2 was actually being detected as AI because the discovery code's binary-target check silently dropped every other generator. The model was a stylegan2 detector hiding behind class-collapsed metrics. Lesson: cross-source check is mandatory; headline accuracy can be a mirage.
- **Run 2 — Discovery fix + stratified sampling, 5K per class.** Found 25 AI sources, distributed evenly. RF hit 71%, SVM 76%. The cross-source check now revealed the *next* problem: lsun (real outdoor scenes) was being classified as AI 72% of the time, and ffhq (real faces) was being routed to AI. Lesson: data composition matters more than data volume.
- **Run 3 — Same fixes, 12.5K per class.** Tried to fix the per-source spread by adding more data. RF accuracy *dropped* 2.4 to 2.7 pp on both backbones. SVM hit 77.62% val and 76.76% test. Lesson: there's a frozen-feature ceiling — more data hurts the high-capacity classifier because the bottleneck is in the representation, not the volume. Bias still visible — landscape photos in the demo predicted "Forged."
- **Run 4 — Mix CASIA Au into Real, 12.5K per class. (Current.)** Real now has 11 sources including CASIA's authentic images, breaking the "this came from CASIA" shortcut. Validation accuracy: 74.92%. Cross-source: 73.27%. The −3 pp from Run 3 is the honest cost of removing the shortcut. Concrete proof: landscape went from ~28% (Run 2) to **89% Real**; lsun to 69%; CASIA Au sits at **49.7% Real** — chance for a binary Real-vs-Forged distinction within CASIA, which is exactly what we want once dataset signature is no longer a free predictor. Forged on Tp/ dropped from 96.5% to 89.08% — same shortcut removal, different angle.

**If asked:** "What was the moment you knew the 84% was fake?"
> First run of step4b. The AI test set was 100% stylegan2 images. That single line of output triggered every fix that followed.

**If asked:** "Was Run 4 worth giving up 3 pp of accuracy?"
> Yes — and the slide-7 audience number (74.92%) is now defensible. The 77.62% from Run 3 was real but inflated; if I cited it I'd have to caveat that landscape photos in the demo were being called Forged. Now the same demo correctly handles them.

---

## Slide 13 — Cross-source sanity check (Step 4b)

**Time:** 75 seconds

**Goal:** Demonstrate the methodological safety net that caught the dataset-bias issues.

**Speaking points:**
- The concern: with Real, Forged, and AI from different datasets and source folders, the classifier could learn dataset signatures (JPEG quantization tables, sensor noise) instead of authenticity cues.
- step4b slices the test set by source folder and reports per-source accuracy and within-class spread.
- Final 12.5K MobileNetV2 + SVM results, after the CASIA Au merge:
  - Forged on CASIA Tp/: **89.08%**, single source.
  - AI: **33% to 100%** across 23 generators, spread of 67.2 pp.
  - Real: **49% to 93%** across 11 sources, spread of 43.8 pp.
  - **CASIA Au: 49.72%** — chance-level, exactly what we want once the shortcut is gone.
- Two specific recoveries:
  - lsun (real scenes) went from 28% (Run 2 with the stylegan2-only bug) to 69% (Run 4). Scene-shortcut largely broken.
  - landscape (real outdoor photos) went from ~28% to 89% over the same arc.
- Built into step4b is a decision rule: spread > 15 pp means **do not fine-tune** the frozen features — it would amplify the existing gap rather than close it. That's the next slide.

**If asked:** "Why did Forged drop 7 points after the Au fix?"
> Because the model used to lean on "this image came from CASIA → Forged." Once Au is also in the Real class, that shortcut is gone. The 89% number reflects the model actually identifying tampering in the image, not the dataset signature.

---

## Slide 14 — Should we fine-tune the CNN? Decision: NO (for now)

**Time:** 75 seconds

**Goal:** Give an explicit, three-reason answer to "why don't you just fine-tune the network?"

**Speaking points:**
- This is the single most-asked question about the project, so we put a slide on it. Three reasons not to fine-tune now.
- **Reason 1: the spread rule says no.** AI spread is 67 pp, Real spread is 44 pp. Both are well over the 15 pp threshold the cross-source check uses for the fine-tuning decision. Fine-tuning amplifies whatever bias exists in the data — easy generators would get easier, hard generators wouldn't move.
- **Reason 2: the data-scaling experiment ruled out "just need more data."** Going from 5K to 12.5K per class lowered RF accuracy on both backbones. So the bottleneck isn't training-set size, and fine-tuning effectively gives the model more capacity to use against the same bottleneck.
- **Reason 3: the hard tier needs a different inductive bias, not a tuned ImageNet net.** StyleGAN3 at 33%, Stable Diffusion at 59%, latent_diffusion at 55%. These generators are designed to look photographic to ImageNet-style nets. The fix is frequency-domain or noise-residual features (NoisePrint, FrequencyNet) — fine-tuning a semantic backbone won't grow features it doesn't already have.
- When *would* we fine-tune? After per-source spread drops below ~15 pp via better dataset balancing, AND only as a targeted last-block tune on Easy + Medium tiers, not generic end-to-end retraining.

**If asked:** "What would change your mind?"
> Two things together: (a) Real and AI within-class spread under 15 pp via balanced dataset sourcing, and (b) initial evidence from a frequency-aware backbone showing Hard-tier improvement. At that point, fine-tuning the last block could lift the Easy + Medium tiers further. Doing it before either of those is solving the wrong problem.

**If asked:** "Doesn't Ali et al. fine-tune and get 92%?"
> Yes, on a binary Real-vs-Forged problem on a single dataset. Their fine-tuning works because the spread isn't an issue at that scope. We're at a point where, on the binary CASIA subtask alone, our frozen system already matches their regime (~89-99% depending on the iteration). The gap is on the 3-class diverse-generator problem, which their setup doesn't address.

---

## Slide 15 — Demo, limitations, future work

**Time:** 90 seconds (plus demo if AV permits)

**Goal:** Live demo, honest limitations, prioritized next steps.

**Speaking points:**
- Demo: `python -m src.step5_gradio_demo` boots a local server. Defaults to MobileNetV2 + SVM + StandardScaler — the best combination. Upload an image, get the predicted class with per-class confidences.
- (Run the demo here. Have 3 sample images preloaded: a real outdoor photo (which now correctly goes to Real after the Au fix — used to fail), a CASIA tampered image, and a hard AI image like Stable Diffusion to demonstrate the limitation honestly.)
- Limitations:
  - Frozen ImageNet features hit a ceiling on modern diffusion. Stable Diffusion sits at ~59%.
  - Random Forest probabilities are uncalibrated — the demo's confidence bars are indicative, not literal.
  - Single seed; no cross-validation.
  - CASIA Au at 49.7% means within-CASIA Real-vs-Forged is at chance. Some Au images still get called Forged due to residual sensor-noise overlap with Tp/ — the shortcut is broken but the residual signal is still there.
- Future work in priority order:
  1. **Frequency-aware backbone** (NoisePrint, FrequencyNet) — addresses the Hard tier directly. (Was item 2 in earlier versions of the deck; promoted because the data-scaling and spread analyses point at the feature space, not data volume or end-to-end fine-tuning.)
  2. **Targeted fine-tune of last block** on Easy + Medium tiers only, after spread drops below 15 pp.
  3. Calibrate RF/SVM probabilities with `CalibratedClassifierCV`.
  4. Leave-one-generator-out evaluation for cross-dataset robustness.
  5. Add more Real-image sources to bring Real spread below the threshold.
- "Thanks — happy to take questions."

**If asked:** "What would you change with another month?"
> Items 1 and 2 in order: bring in a frequency-aware backbone first (closes the Hard-tier gap), then targeted last-block fine-tuning on the Easy + Medium tiers (lifts the headline number). Doing fine-tuning first wastes effort on the wrong inductive bias.

---

## General Q&A bank

**Q: How long does the full pipeline take?**
> About 3.5 hours end-to-end at 12,500 samples per class on CPU: ~25 min for image processing in step 1, ~70 min for Xception feature extraction plus ~32 min for MobileNetV2, ~45 min for SVM training, fast for everything else. The 5K version is about 75 minutes total.

**Q: Why Xception specifically?**
> It's the backbone used by Ali et al. and several other published forgery detectors, so we get apples-to-apples on architecture. It's also a strong general-purpose feature extractor at ~22M parameters — small enough to run on CPU.

**Q: What was the hardest part?**
> Honestly, the dataset wrangling. ArtiFact's 25 generator folders use two different metadata schemas, and the discovery code that handles this is more complex than the actual model code. Plus the methodological iteration loop — we ran the full pipeline four times before the cross-source check stopped surfacing new bugs.

**Q: Could this run on a phone?**
> The MobileNetV2 ablation is specifically there to answer that. MobileNetV2 + SVM is the winning combination, so the same pipeline can transfer to mobile inference at a fraction of the cost. The classifier is small enough to ship with the model.

**Q: Adversarial robustness?**
> Out of scope. The pipeline is not adversarially robust — a deliberately crafted attack image would likely fool it. Real production systems combine this kind of detector with provenance signals (C2PA metadata, watermarks) for that reason.

**Q: Did you test cross-dataset transfer?**
> Step 4b is the in-distribution version of that test — same datasets, different source folders. Full leave-one-generator-out (train without one generator, test on it) is in future work — it's the strongest test but requires retraining N times for N generators.

**Q: Did you try logistic regression?**
> Briefly, in early iterations. On dense GAP features, logistic regression typically competes with RBF SVM at 10x the training speed. We kept SVM as the headline because it's what Ali et al. used. Logistic regression with calibration would be the right pick for a production deployment.

**Q: Why did the val accuracy go DOWN between Run 3 and Run 4?**
> That's the central methodology point on slide 12. Run 3's 77.62% had a dataset-source shortcut baked in — the model could predict "Forged" by recognizing CASIA-style images. Run 4 added CASIA Au into the Real class, so that shortcut is gone, and the model has to actually classify by image content. The 3-point drop is honest cost; the new 74.92% reflects the actual ability of frozen ImageNet features at this 3-class task.

**Q: Will fine-tuning fix the Hard tier?**
> No, and that's the point of slide 14. Fine-tuning a semantic-feature net amplifies what it already does well. The Hard tier (StyleGAN3, modern diffusion) fails because those generators are designed to look photographic to ImageNet features — there's nothing in the feature space to amplify. The right fix is a different backbone (frequency-aware), not more training on the wrong one.
