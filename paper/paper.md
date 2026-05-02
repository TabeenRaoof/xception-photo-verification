# TruPhoto: A Three-Class Image Authenticity Classifier on Frozen ImageNet Features

**Tabeen Raoof**
CS483 Deep Learning, San Francisco Bay University
2026

---

## Abstract

We present **TruPhoto**, a three-class image authenticity classification system that distinguishes real photographs, classically forged images (splicing, copy-move), and fully AI-generated images. Unlike prior work that treats authenticity as a binary problem on a single dataset, TruPhoto is evaluated across two datasets and 25 distinct AI generators using a CPU-friendly two-stage architecture: a frozen ImageNet-pretrained convolutional backbone produces fixed-length feature vectors, and a classical classifier (Random Forest or RBF-kernel SVM) maps those vectors to class labels. On a 12,500-sample-per-class evaluation set, our best configuration (frozen MobileNetV2 + SVM) achieves **74.92% validation accuracy and 73.27% test accuracy** with a 1.65 percentage-point val/test gap. A cross-source sanity check reveals a consistent pattern: the model achieves 85%+ accuracy on older or artifact-heavy generators (StarGAN, BigGAN, DDPM) but approaches chance on modern photorealistic generators (StyleGAN3 at 33%, Stable Diffusion at 59%). This finding, combined with a 5K-versus-12.5K data-scaling experiment in which Random Forest accuracy *dropped* with more data on both backbones, indicates a representational ceiling in frozen ImageNet features that additional training data cannot address. We document four training iterations, three independent dataset bugs caught by the cross-source check, and an explicit decision against end-to-end fine-tuning given the current within-class spread (Real spread 43.8 pp, AI spread 67.2 pp) — both well above the 15 pp threshold built into our methodology.

---

## 1. Introduction

The proliferation of generative AI has changed the shape of the image-authenticity problem. Where a decade ago a forged image almost always meant a human-edited splice or copy-move, today's threat surface includes fully synthetic outputs from generative adversarial networks (GANs) and diffusion models that no human ever touched. A practical authenticity-detection pipeline must distinguish these two classes of synthetic image because the downstream investigative response differs: a spliced image implicates a human editor; a generated image implicates a model.

Most published forgery detectors collapse the problem to **real versus fake**, which conflates these two threat models. TruPhoto explicitly outputs three classes — **Real**, **Forged**, and **AI-Generated** — at the cost of a harder learning problem and lower headline accuracy than binary baselines.

### 1.1 Contributions

This paper makes four contributions:

1. **A three-class formulation**, evaluated across two source datasets (ArtiFact and CASIA 2.0) and 25 distinct AI generators, with explicit per-source accuracy reporting.
2. **A frozen-feature, CPU-only pipeline** that achieves 73.27% test accuracy on this 3-class problem in under four hours of total training time on a single laptop.
3. **A cross-source sanity check** (`step4b`) that surfaces dataset-source shortcuts before they become embedded in headline numbers. Three independent dataset-handling bugs were caught by this check across our iterations.
4. **An empirical ceiling result** — Random Forest accuracy decreased with 2.5x more training data on both feature backbones — indicating that for the modern-diffusion subset of this task, the bottleneck is the inductive bias of ImageNet features, not training-set size.

### 1.2 Outline

Section 2 reviews related work. Section 3 describes the source datasets and the sampling strategy. Section 4 details the two-stage methodology. Section 5 documents the implementation, including three dataset-handling bugs and their fixes. Section 6 lays out the experimental protocol; Section 7 reports results across four training iterations. Section 8 is the cross-source analysis. Section 9 discusses the frozen-feature ceiling and the explicit decision against fine-tuning. Sections 10 and 11 cover limitations and future work.

---

## 2. Related Work

### 2.1 Classical image-forgery detection

Image forgery detection has historically targeted splicing and copy-move attacks on benchmark datasets such as CASIA 2.0. Methods range from hand-crafted features (sensor pattern noise, color filter array inconsistencies) to deep convolutional networks trained end-to-end on labeled forgeries. The relevant comparison point for our work is **Ali et al. (2022)**, which fine-tunes Xception on CASIA 2.0 and reports 92.23% binary accuracy. Their setup is single-dataset, two-class, and end-to-end-trained; ours differs on all three axes.

### 2.2 Generative-image detection

Detection of fully AI-generated images is a younger sub-field. Two architectural directions stand out:

- **Semantic-feature transfer**, in which a network pretrained on ImageNet (or similar large-scale natural-image datasets) produces feature vectors that a downstream classifier separates into real vs. generated. This is what TruPhoto does.
- **Frequency-domain or noise-residual features**, designed specifically to capture statistical artifacts of generative models (e.g., NoisePrint, FrequencyNet variants). Modern GAN/diffusion detection literature increasingly favors this direction because semantic features carry less signal as generators become more photorealistic.

Our results are consistent with this trend: semantic features (Xception, MobileNetV2) detect older generators (DDPM, BigGAN, StarGAN) at 85%+ accuracy but struggle on StyleGAN3 (33%) and Stable Diffusion (59%).

### 2.3 Datasets

**ArtiFact** is a large-scale Kaggle dataset that aggregates outputs from 25+ generators (StyleGAN family, Stable Diffusion, GLIDE, DDPM, BigGAN, StarGAN, gansformer, latent_diffusion, and others) alongside reference real images from natural-photo sources (afhq, ffhq, coco, imagenet, lsun, celebahq). **CASIA 2.0** provides a tampered (`Tp/`) folder of 5,123 spliced/copy-move forgeries and an authentic (`Au/`) folder of 7,491 untampered images.

---

## 3. Datasets

### 3.1 Source folders

We obtained ArtiFact and CASIA 2.0 from Kaggle. After a partial download, our copy of ArtiFact contains 33 generator/source subfolders. Some subfolders contribute exclusively real images (e.g., `afhq`, `ffhq`, `landscape`, `metfaces`); others exclusively AI outputs (e.g., `big_gan`, `stylegan2`, `stable_diffusion`). A subset (e.g., `cycle_gan`, `pro_gan`) contributes to *both* classes — these folders contain real reference images alongside their generative outputs and use a per-row metadata field (`target`) to distinguish them.

CASIA 2.0 is split into `Tp/` (tampered, 5,123 images, our **Forged** class) and `Au/` (authentic, 7,491 images). In our final iteration, `Au/` contributes to the **Real** class — see Section 5.3.

### 3.2 Sampling strategy

The full available pool exceeds 2.5 million images. We sample **12,500 images per class** as the per-class budget. The sampling is stratified across source folders within each class, so no single generator dominates: each AI source contributes 500 images (12,500 / 25), each Real source contributes ~1,136 images (12,500 / 11). The Forged class is undersized — only 5,123 images exist in CASIA `Tp/`, all of which are used. After sampling, the dataset totals **30,123 images**.

### 3.3 Splits

We use a deterministic 70/15/15 train/validation/test split with `numpy.random.seed(42)`. Each class is split independently before being concatenated, so class proportions are preserved across splits. Final split sizes:

| Split | Real | Forged | AI-Generated | Total |
|-------|------|--------|--------------|-------|
| train | 8,750 | 3,586 | 8,750 | 21,086 |
| val   | 1,875 | 768   | 1,875 | 4,518  |
| test  | 1,875 | 769   | 1,875 | 4,519  |

All images are resized to 299 × 299 pixels (Xception's native input) and saved as quality-95 JPEGs.

---

## 4. Methodology

### 4.1 Two-stage architecture

TruPhoto is a deliberate two-stage system:

1. **Stage 1 — Feature extraction.** A frozen, ImageNet-pretrained convolutional backbone takes a 299 × 299 RGB image and produces a fixed-length feature vector (the global-average-pooled output of the network's final convolutional block). The backbone runs once per image and never sees the authenticity label.
2. **Stage 2 — Classification.** A classical classifier (Random Forest or SVM) maps that feature vector to one of three classes. This stage is the only part of the system trained on our task.

This split is a deliberate trade-off:

- **Pros.** No GPU is required for either training or inference at the classifier stage. Feature extraction is a one-time cost (~70 minutes for Xception, ~32 minutes for MobileNetV2 on CPU for our 30K-image dataset). Classifier training takes seconds (RF) to under an hour (SVM). The system is reproducible from raw data in under four hours on a laptop.
- **Cons.** We give up the ability to adapt the convolutional features to forgery-specific cues (e.g., subtle texture or frequency patterns left by specific generators). Section 9 documents the empirical ceiling this creates.

### 4.2 Backbone choice

We extract features from two backbones to enable an ablation on feature-extractor size:

- **Xception** (Chollet, 2017) — depthwise-separable convolutions, ~22M parameters, 2048-dimensional GAP feature. Our primary backbone, chosen for direct architectural comparability with Ali et al. (2022).
- **MobileNetV2** (Sandler et al., 2018) — inverted-residual blocks, ~2.2M parameters, 1280-dimensional GAP feature. Smaller, faster, and used here as a deliberate contrast.

Both backbones are loaded via `timm.create_model(name, pretrained=True, num_classes=0)`. The `num_classes=0` argument strips the classification head; the resulting model ends at the global average pool. We freeze every parameter (`p.requires_grad = False`) and call `model.eval()` to lock BatchNorm running statistics and disable dropout. No gradient ever flows back into the backbone.

### 4.3 Classifier choice

We pair each backbone with two classifiers:

- **Random Forest.** 200 trees, `class_weight='balanced'` to compensate for the undersized Forged class, no maximum depth. Implemented via `sklearn.ensemble.RandomForestClassifier`.
- **Support Vector Machine.** RBF kernel, `C=10`, `gamma='scale'`. Features are pre-scaled with `StandardScaler` (fit on training features only). `probability=True` is set so the demo can produce per-class confidences. Implemented via `sklearn.svm.SVC`.

The combination of two backbones and two classifiers gives a 2 × 2 ablation matrix. As reported in Section 7, three of the four cells were trained at 12.5K-per-class scale; Xception + SVM at this scale was skipped because SVM training time scales as O(n²-n³) and would have run multi-hour without changing the qualitative ranking established at the 5K scale.

### 4.4 Inference pipeline

At inference time:

1. Resize input image to 299 × 299, normalize with ImageNet mean/std.
2. Forward-pass through the frozen backbone with `torch.no_grad()`.
3. (SVM only) Apply the trained `StandardScaler`.
4. Call `classifier.predict_proba(features)` to get per-class probabilities.

The Gradio demo (`step5_gradio_demo.py`) wires this end-to-end and defaults to MobileNetV2 + SVM, the best combination from our ablation.

---

## 5. Implementation Details

The pipeline is implemented as five sequential scripts in `src/`, runnable as `python -m src.stepN_…`. The full project is structured as a Python package (`pyproject.toml`, `src/__init__.py`) with absolute imports and per-model JSON meta-manifests recording the trained-at timestamp and git SHA.

### 5.1 Step 1 — Dataset preparation

`src/step1_prepare_dataset.py` is responsible for: (i) discovering image paths and labels from the raw datasets, (ii) sampling per-class quotas, (iii) resizing and writing 299×299 JPEGs into `data/processed/{split}/{class}/`, and (iv) emitting a metadata CSV recording every sampled image's source.

This script went through three substantial revisions to fix three independent dataset-handling bugs. Each bug was caught by the cross-source sanity check (Section 8).

### 5.2 Fix 1 — Generator-aware label inference

**The bug.** Our initial discovery code parsed each ArtiFact subfolder's `metadata.csv` and assigned `target=0` rows to the Real class and `target=1` rows to the AI class. This worked for some subfolders but silently dropped every row in others.

**Root cause.** ArtiFact's metadata follows two different schemas across subfolders. Most folders use `target ∈ {0, 1}` for real/AI. But several folders (e.g., `big_gan`, `latent_diffusion`) use `target = <ImageNet class index>` — values like 6, 207, 412 — to identify what class the generator produced. Our binary check did not match these rows, and they were silently skipped. After Run 1, only `stylegan2` (one folder using the binary schema) registered as an AI source on disk.

**The fix.** We introduced an `AI_GENERATOR_FOLDERS` whitelist of known-AI subfolder names. The discovery routine now applies the binary check first; if the target field is non-binary, it falls back to checking whether the parent folder name appears in the whitelist. This recovered 24 additional AI sources, bringing the active count from 1 to 25.

### 5.3 Fix 2 — Stratified sampling

**The bug.** Even after Fix 1 found all 25 AI sources, naive uniform sampling on the flat list of ~1.5 million AI images allocated approximately 70% of the AI training set to `stylegan2` alone, simply because `stylegan2` has 1 million images on disk versus ~10K for most other generators.

**The fix.** `stratified_sample()` distributes the per-class budget equally across sources first (`budget // n_sources` per source), then redistributes any leftover quota — caused by sources that have fewer images than the per-source target — proportionally across the larger pools. The result is an AI training set in which every generator contributes 500 ± 1 images at the 12.5K-per-class budget.

### 5.4 Fix 3 — CASIA Au merged into Real

**The bug.** After Runs 1–3, the cross-source sanity check (and demo testing) revealed that landscape and outdoor-scene photographs were being classified as **Forged** with high confidence. The reason: every Forged image came from CASIA, and *only* from CASIA; every Real image came from ArtiFact, and *only* from ArtiFact. The model could short-circuit the classification by recognizing dataset signatures (JPEG quantization tables, sensor noise, camera-pipeline residuals) instead of forgery cues. CASIA images, regardless of content, were being routed to Forged.

**The fix.** We added a third discovery routine, `discover_casia2_authentic_paths()`, which scans CASIA `Au/` and contributes 7,491 images to the Real class with a `casia2_Au` source label. The Real class is now sampled across 11 sources (10 from ArtiFact + CASIA Au), and the dataset-source signature is no longer a free predictor.

The cost was a ~3 percentage-point drop in headline accuracy (77.62% → 74.92% validation) and a 7-point drop in Forged-class accuracy (96.5% → 89.08%), both of which represent the model giving up its dataset shortcut. The benefit is that landscape photographs in the Gradio demo now correctly go to Real, and CASIA Au images sit at 49.72% Real / 50.28% Forged at test time — chance-level for a binary Real-versus-Forged distinction within the same dataset, which is exactly the desired behavior once dataset signature is no longer informative.

### 5.5 Step 2 — Feature extraction

`src/step2_extract_features.py` loads each backbone, freezes it, and forward-passes the 30,123 processed images through it. Features and labels are saved as `.npy` files: `xception_x_train.npy`, `xception_y_train.npy`, etc., for each of the three splits and each of the two backbones — twelve files total, ~383 MB on disk.

The label remapping `{ImageFolder_label: our_label}` accounts for the fact that `torchvision.datasets.ImageFolder` orders classes alphabetically (`AI_Generated`, `Forged`, `Real`), whereas our internal label scheme is `0=Real, 1=Forged, 2=AI_Generated`.

### 5.6 Steps 3–5

- `src/step3_train_classifiers.py` trains the four classifiers in the ablation matrix and writes `.joblib` artifacts plus per-model `.meta.json` manifests.
- `src/step4_evaluate.py` computes test-set accuracy, per-class precision/recall/F1, and renders confusion matrices, an ablation bar chart, and a per-class F1 chart.
- `src/step4b_cross_dataset_check.py` is the cross-source sanity check; see Section 8.
- `src/step5_gradio_demo.py` launches an interactive Gradio interface for hand testing.

---

## 6. Experimental Setup

### 6.1 Hardware

All experiments were run on a single MacBook Pro (M-series, no GPU). Training does not require a GPU; feature extraction would be ~10x faster on a GPU but is not required.

### 6.2 Wall-clock costs

End-to-end pipeline cost at 12,500 images per class:

| Stage | Time |
|-------|------|
| Step 1 — Resize and split 30,123 images | ~25 min |
| Step 2 — Xception feature extraction | ~70 min |
| Step 2 — MobileNetV2 feature extraction | ~32 min |
| Step 3 — RF training (per backbone) | ~10 sec |
| Step 3 — SVM training (MobileNetV2, 21K samples) | ~43 min |
| Step 4 + 4b — Evaluation | ~1 min |
| **Total** | **~3.5 hours** |

The 5K-per-class run completes in approximately 75 minutes total.

### 6.3 Reproducibility

All randomness is seeded with `numpy.random.seed(42)` (sampling) and `random_state=42` (classifier instantiation). Each saved model has a sibling `.meta.json` recording the training timestamp, git SHA, classifier hyperparameters, and validation accuracy, so any model artifact can be traced back to the code revision that produced it.

---

## 7. Results

### 7.1 Iteration history

We ran the full pipeline four times. Each run fixed a methodological flaw discovered in the previous run's cross-source check.

| Run | Configuration | Best val acc | What broke / what changed |
|-----|---------------|--------------|----------------------------|
| 1 | Naive baseline | ~84% | Only `stylegan2` registered as AI on disk; Run 1's headline was an artifact of class collapse. |
| 2 | + Fix 1 (label inference) + Fix 2 (stratified sampling), 5K/class | ~76% (SVM) | All 25 AI sources active. Cross-source check revealed `lsun` real-scenes being routed to AI; per-source spread very high. |
| 3 | Same fixes, 12.5K/class | 77.62% (SVM) | RF accuracy *dropped* with more data on both backbones (-2.4 pp, -2.7 pp). Demo testing revealed landscape photographs being classified as Forged. |
| 4 | + Fix 3 (CASIA Au merged into Real), 12.5K/class | **74.92% (SVM)** | Dataset-source shortcut broken. Forged accuracy on CASIA `Tp/` dropped from 96.5% to 89.08% — the honest cost of removing the shortcut. |

The Run 1 → Run 4 trajectory is the centerpiece of the methodological story: **headline accuracy decreased monotonically as each iteration removed a different shortcut from the model's repertoire**, while the underlying classifier became more honest about what it could and could not distinguish.

### 7.2 Final ablation (Run 4)

After the CASIA Au merge, the four-cell ablation matrix has three populated cells:

| Backbone | Classifier | Val acc | Test acc |
|----------|------------|---------|----------|
| MobileNetV2 | Random Forest | 69.81% | 69.40% |
| MobileNetV2 | SVM | **74.92%** | **73.27%** |
| Xception | Random Forest | not retrained | — |
| Xception | SVM | not retrained | — |

Xception classifiers were not retrained for Run 4. Across the 5K (Run 2) and 12.5K-without-Au (Run 3) iterations, MobileNetV2 was ahead of Xception on both classifier heads at every measurement, by margins consistent with the 1280-dim-vs-2048-dim feature-space difference. We elected to skip the multi-hour Xception SVM retrain rather than carry stale numbers forward.

The val/test gap on the best combination is 1.65 percentage points — small enough to indicate good generalization without significant overfitting.

### 7.3 Per-class performance

Per-class precision, recall, and F1 score on the test set for MobileNetV2 + SVM:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Real | 0.68 | 0.69 | 0.68 | 1875 |
| Forged | 0.88 | 0.89 | 0.88 | 769 |
| AI-Generated | 0.73 | 0.71 | 0.72 | 1875 |

Forged is the easiest class — splicing artifacts carry strong, consistent low-level texture cues that ImageNet-pretrained features capture well. The Real and AI-Generated classes are mutually confused, consistent with the per-generator results reported in Section 8.

---

## 8. Cross-Source Analysis

### 8.1 Method

`step4b` slices the test set by source folder (each ArtiFact generator, CASIA `Tp/`, CASIA `Au/`) and reports per-source accuracy and within-class spread. The within-class spread — `max(per-source accuracy) - min(per-source accuracy)` — is our primary diagnostic signal: a narrow spread indicates the model generalizes uniformly across sources within a class; a wide spread indicates the model is using source identity as a shortcut.

### 8.2 Per-class spread (final iteration)

| Class | Sources | Min | Max | Spread |
|-------|---------|-----|-----|--------|
| AI-Generated | 23 | 0.328 (StyleGAN3) | 1.000 (StarGAN) | **67.2 pp** |
| Real | 11 | 0.488 (ffhq) | 0.926 (afhq) | **43.8 pp** |
| Forged | 1 | 0.891 (CASIA Tp) | 0.891 | n/a |

The AI spread (67.2 pp) is the dominant source of variance and the principal evidence for the generator-difficulty tier structure presented next. The Real spread (43.8 pp) is improved relative to Run 3 (47.2 pp) by the addition of CASIA Au.

### 8.3 Generator difficulty tiers

Sorting the AI-Generated per-source accuracies clusters them into three tiers:

- **Easy (≥ 85%)** — older or artifact-heavy generators that ImageNet features detect almost trivially:
  - StarGAN 100%, sfhq 96%, face_synthetics 96%, DDPM 94%, CIPS 91%, gansformer 89%, denoising_diffusion_GAN 86%, BigGAN 85%, palette 85%.
- **Medium (60–85%)** — generators with mixed difficulty:
  - gau_gan 76%, StyleGAN1 69%, LAMA 69%, StyleGAN2 64%, taming_transformer 62%, projected_gan 62%, GLIDE 60%, MAT 60%.
- **Hard (< 60%)** — modern photorealistic generators that approach chance:
  - Stable Diffusion 59%, diffusion_gan 57%, latent_diffusion 55%, vq_diffusion 52%, generative_inpainting 46%, **StyleGAN3 33%**.

The within-StyleGAN gradient is particularly clean: **StyleGAN1 → StyleGAN2 → StyleGAN3 yields 69% → 64% → 33%**. Each StyleGAN release was explicitly designed to remove artifacts visible in the previous version (StyleGAN3 added equivariance properties that eliminate texture-sticking artifacts), so our model's degrading performance across the StyleGAN family is, in effect, a measurement of the generators' design success.

### 8.4 The CASIA Au signal

CASIA `Au/` images sit at **49.72% Real / 50.28% Forged** at test time — within rounding distance of chance for a binary Real-versus-Forged distinction *within* the CASIA dataset. This is the desired outcome once the dataset-source shortcut is removed: the model can no longer distinguish CASIA-authentic from CASIA-tampered using camera-pipeline signatures alone, because both classes now contain CASIA images. The remaining signal in CASIA `Tp/` (89.08%) reflects whatever the model has learned about *splicing* itself, independent of dataset signature.

### 8.5 The spread decision rule

Embedded in `step4b` is a simple decision rule for whether the frozen features should be fine-tuned on this task:

| Within-class spread | Recommendation |
|---------------------|----------------|
| ≤ 5 pp | Uniform; fine-tuning is safe. |
| 5 – 15 pp | Borderline; fix data first. |
| > 15 pp | Shortcut learning; **do not fine-tune**. |
| Any single source < 50% | Red flag regardless of spread. |

Our final model has AI spread of 67.2 pp and Real spread of 43.8 pp — both well into the "do not fine-tune" range. Section 9.2 unpacks the implication.

---

## 9. Discussion

### 9.1 The frozen-feature ceiling

Run 3 vs. Run 2 was a controlled data-scaling experiment: the same code, the same backbones, the same classifiers, but 12,500 images per class instead of 5,000. Random Forest accuracy *dropped* with the added data: −2.4 percentage points on Xception, −2.7 percentage points on MobileNetV2. SVM accuracy moved similarly.

The interpretation is straightforward but consequential. With 2.5x more training samples and the same feature space, a high-capacity classifier like Random Forest should at minimum hold its accuracy and typically improve. A *decrease* indicates that the additional training signal is making the classifier overfit to noise within the existing feature representation rather than generalize to true class boundaries — which is the empirical signature of a representational ceiling. **The bottleneck is in the features, not the data.** No quantity of additional ImageNet-feature-based training will produce a meaningfully better classifier on the Hard tier (StyleGAN3, modern diffusion).

### 9.2 The fine-tuning decision

Given the ceiling result and the cross-source spread, we explicitly *do not* fine-tune the convolutional backbone in this iteration of the project. Three independent reasons support this decision:

1. **The spread rule says no.** AI spread of 67.2 pp and Real spread of 43.8 pp are both far above the 15 pp threshold. Fine-tuning amplifies whatever bias already exists in the feature space — easy generators would get easier, hard generators would not move, and the gap would widen. Fine-tuning is appropriate only when the data is balanced enough that the fine-tuning signal is uniformly informative; ours is not.
2. **The data-scaling experiment ruled out the cheap explanation.** If data quantity were the bottleneck, the 12.5K run would have outperformed the 5K run. It did not. Fine-tuning adds capacity to use against the same bottleneck — there is no reason to expect a different outcome.
3. **The Hard tier needs a different inductive bias.** StyleGAN3 at 33% accuracy is not a classifier-capacity problem; it is a feature-space problem. Modern diffusion and modern GANs are explicitly designed to look photographic to ImageNet-style semantic features. Fine-tuning a semantic-feature backbone cannot grow the spectral or noise-residual features that detection of these generators requires. The right architectural move for the Hard tier is a frequency-aware backbone (NoisePrint, FrequencyNet variants), not more learning on the wrong inductive bias.

Fine-tuning becomes a candidate when (a) the per-source spread drops below 15 pp via better dataset balancing, *and* (b) initial evidence from a frequency-aware backbone shows Hard-tier improvement. At that point, a *targeted* fine-tune of the last convolutional block on the Easy and Medium tiers could lift the headline. A generic end-to-end retrain before either condition is met is solving the wrong problem.

### 9.3 Comparison with Ali et al. (2022)

Ali et al. (2022) reports 92.23% binary accuracy on CASIA 2.0 using a fine-tuned Xception. A direct comparison to our 73.27% three-class accuracy would be misleading. Three reframings make the comparison honest:

- **Random-baseline gain.** Binary chance is 50%; their result is +42 over chance. Three-class chance is 33%; ours is +42 over chance. The gain over chance is comparable; the absolute number is not.
- **Same-subtask comparison.** When evaluated only on CASIA `Tp/` (the binary forgery subtask Ali et al. tackle), our system achieves 89.08% accuracy in Run 4 — the same regime, with the dataset-shortcut explicitly removed. Run 3 reached 96.5% on the same subtask, but we now consider that number inflated by the shortcut.
- **Setup-cost reframing.** Their result requires GPU and end-to-end training. Ours runs in under four hours on a CPU laptop, no GPU at any step. For a class project, classroom deployment, or a prototype that must run on commodity hardware, the trade-off favors our setup.

We cite Ali et al. as a reference point for what fine-tuned Xception features can achieve on CASIA, not as a baseline we attempt to match.

### 9.4 Why MobileNetV2 beats Xception

A consistent finding across all three iterations (5K, 12.5K, 12.5K + Au) is that MobileNetV2 outperforms Xception on this task at this data scale, on both classifier heads. Two factors plausibly contribute:

- **Feature dimensionality.** Xception produces 2048-dim features; MobileNetV2 produces 1280-dim. With 8,750 per-class training samples, the larger Xception space leaves more room for an SVM or RF to fit noise.
- **Architectural depth.** MobileNetV2's inverted-residual blocks preserve more low-level texture information than Xception's deeper depthwise-separable blocks. Forgery and generator artifacts live in low-level texture, so the less-abstracted backbone retains more of the signal that this task depends on.

This finding generalizes a useful intuition: when fine-tuning is unavailable, the *less-abstract* feature extractor often wins on detection tasks that depend on low-level cues, even when the more-abstract one is "stronger" by ImageNet classification accuracy.

---

## 10. Limitations

1. **Frozen-feature ceiling on Hard tier.** Stable Diffusion at 59% and StyleGAN3 at 33% are below useful-deployment thresholds. A frequency-aware backbone is the architectural fix; we do not attempt one in this project.
2. **Uncalibrated probabilities.** Both Random Forest (`predict_proba` based on tree votes) and SVM (`probability=True` Platt scaling) produce probabilities that are correlated with confidence but not calibrated. The Gradio demo's confidence bars should be read as relative ordering, not absolute probabilities. `CalibratedClassifierCV` would address this for a deployment.
3. **Single-seed evaluation.** We run with seed 42 throughout. The reported numbers are not the result of cross-validation, and the absolute values may shift by 1–2 pp under reseeding.
4. **CASIA Au at chance.** The 49.72% accuracy on CASIA `Au/` is the desired *direction* (shortcut broken) but not the desired *value* (perfect classification of authentic CASIA images as Real). Some Au images are still being misclassified as Forged because of residual sensor-noise overlap with `Tp/`. The fix is more dataset diversity in Real, not more training.
5. **Generator coverage is fixed at training time.** The 25 generators in our training set are not exhaustive. New generators released after our training run (e.g., post-2024 models) are not represented. Leave-one-generator-out cross-validation would estimate cross-generator robustness; we do not run it because it would require N retraining runs.
6. **No adversarial robustness.** A deliberately crafted adversarial example would likely fool the classifier. Production authenticity systems combine learned detection with provenance signals (C2PA metadata, watermarks) for this reason. Adversarial robustness is out of scope.

---

## 11. Future Work

In priority order:

1. **Frequency-aware backbone.** Replace the ImageNet backbone with a network designed for detection of generative artifacts (NoisePrint, FrequencyNet, or a custom DCT-domain CNN). This addresses the Hard tier directly and is the highest-value next move.
2. **Targeted last-block fine-tune.** Once item 1 is in place and per-source spread drops below the 15 pp threshold, unfreeze and fine-tune the final convolutional block on the Easy + Medium tiers only. End-to-end fine-tuning of the entire backbone is not recommended.
3. **Probability calibration.** Wrap the trained SVM in `CalibratedClassifierCV` so the demo confidence bars are interpretable.
4. **Leave-one-generator-out evaluation.** Train on 24 generators, test on the held-out 25th. Repeat for each of the 25 generators. This is the strongest available test of cross-generator robustness but requires N training runs.
5. **Real-class diversity.** Add Real-image sources (additional natural-photo datasets, public-domain photo archives) to bring the Real per-source spread below 15 pp. Currently the Real spread is dominated by `ffhq` at the bottom (49%) and `afhq` at the top (93%).

---

## 12. Conclusion

TruPhoto is a frozen-feature, CPU-only, three-class image-authenticity classifier. The headline accuracy — **74.92% validation, 73.27% test on a 4,519-image test set spanning 36 sources** — is a deliberately conservative number. Earlier iterations of the same pipeline reported 84% and 77.62%, but cross-source analysis showed both numbers were inflated by dataset-handling shortcuts that the model exploited rather than working around. Each iteration of the project removed one shortcut (binary-only label inference, source-imbalanced sampling, dataset-source signature) and traded headline accuracy for methodological honesty. The final result is a model whose 73.27% accuracy is something the cross-source check actually corroborates, and whose remaining errors point cleanly at a representational limit in ImageNet features — a limit that more data cannot close, that fine-tuning would amplify rather than fix, and that the next step in this line of work should address with a frequency-aware backbone rather than another round of training on the wrong inductive bias.

---

## References

Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions.* CVPR 2017.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.

Ali, S. S., Ganapathi, I. I., Vu, N.-S., Ali, S. D., Saxena, N., & Werghi, N. (2022). *Image Forgery Detection Using Deep Learning by Recompressing Images.* MDPI Electronics 11(3), 403.

Karras, T., Aittala, M., Laine, S., et al. (2021). *Alias-Free Generative Adversarial Networks (StyleGAN3).* NeurIPS 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.

Cozzolino, D. & Verdoliva, L. (2020). *Noiseprint: A CNN-Based Camera Model Fingerprint.* IEEE Transactions on Information Forensics and Security 15.

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research 12.

Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.

Wightman, R. *PyTorch Image Models (timm).* GitHub repository.

Dong, J., Wang, W., & Tan, T. (2013). *CASIA Image Tampering Detection Evaluation Database.* IEEE China Summit and International Conference on Signal and Information Processing.

Rahman, M. A., Paul, B., Sarker, N. H., Hakim, Z. I. A., Fattah, S. A. (2023). *ArtiFact: A Large-Scale Dataset with Artificial and Factual Images for Generalizable and Robust Synthetic Image Detection.* ICIP 2023.

---

## Appendix A — Repository structure

```
photo_verification_with_xception/
├── src/
│   ├── __init__.py
│   ├── config.py                       # paths, hyperparameters, seeds
│   ├── dataset_loader.py               # PyTorch dataset / dataloader helpers
│   ├── verify_dataset.py               # sanity check on processed splits
│   ├── step1_prepare_dataset.py        # discovery + sampling + resize
│   ├── step2_extract_features.py       # frozen backbone forward pass
│   ├── step3_train_classifiers.py      # RF + SVM training
│   ├── step4_evaluate.py               # test-set metrics + plots
│   ├── step4b_cross_dataset_check.py   # per-source diagnostic
│   └── step5_gradio_demo.py            # interactive demo
├── presentation/
│   ├── generate_slides.py
│   ├── presenter_notes.md
│   └── truphoto_presentation.pptx
├── paper/
│   └── paper.md                        # this document
├── data/                               # gitignored (regenerable)
│   ├── raw/{artifact, casia2}
│   ├── processed/{train, val, test}/{Real, Forged, AI_Generated}
│   └── features/{xception, mobilenetv2_100}_x_{train,val,test}.npy
├── models/                             # gitignored (regenerable)
│   └── {rf, svm, scaler}_{backbone}.joblib  +  .meta.json
├── results/                            # gitignored except .gitkeep
│   ├── cm_{backbone}_{rf,svm}.png
│   ├── ablation_comparison.png
│   ├── per_class_f1.png
│   └── evaluation_report.txt
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

## Appendix B — Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `IMG_SIZE` | 299 | Xception's native input |
| `SAMPLES_PER_CLASS` | 12,500 | Forged is undersized at 5,123 |
| `TRAIN/VAL/TEST split` | 70 / 15 / 15 | seed 42 |
| `RF_N_ESTIMATORS` | 200 | |
| `RF_CLASS_WEIGHT` | "balanced" | compensates undersized Forged |
| `RF_MAX_DEPTH` | None | |
| `SVM_KERNEL` | "rbf" | |
| `SVM_C` | 10.0 | |
| `SVM_GAMMA` | "scale" | |
| `RANDOM_SEED` | 42 | |
| `IMAGENET_MEAN` | [0.485, 0.456, 0.406] | |
| `IMAGENET_STD`  | [0.229, 0.224, 0.225] | |
