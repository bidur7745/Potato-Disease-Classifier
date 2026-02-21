# Final Year Project Documentation

## Title
**Robust Leaf Disease Detection System Using Cascaded Deep Learning Models with Out-of-Distribution Handling**

---

## 1. Introduction

In real-world agricultural applications, image-based plant disease detection systems are exposed to a wide variety of uncontrolled inputs. Users may upload irrelevant images such as posters, indoor objects, blurred photos, screenshots, or non-plant images. Traditional deep learning classifiers operate under a *closed-set assumption*, where every input is forced into one of the known classes. This leads to highly confident but incorrect predictions, which is unacceptable in agriculture-related decision-support systems.

To address this challenge, this project adopts an **industry-aligned, safety-aware system design** that explicitly handles invalid and out-of-distribution (OOD) inputs. The proposed solution uses a **cascaded (two-stage) deep learning architecture** to ensure robustness, reliability, and real-world deployability.

---

## 2. Problem Statement

Most plant disease classification models:
- Assume all input images belong to predefined disease classes
- Lack mechanisms to reject invalid or unknown images
- Produce overconfident predictions on irrelevant inputs

This results in false positives, reduced trust, and unsafe recommendations. The objective of this project is to design a system that can:
- Identify whether an uploaded image is a plant leaf
- Reject invalid or non-leaf images
- Accurately classify diseases only when the input is valid

---

## 3. System Overview

The system is divided into two main components:

1. **Training Phase (Model Development – Jupyter Notebook)**
2. **Deployment Phase (Inference & Validation – API Server)**

A cascaded inference pipeline is used, where each model has a clearly defined responsibility.

---

## 4. Cascaded Model Architecture

### 4.1 High-Level Pipeline

```
User Upload Image
        ↓
Image Quality & Format Validation
        ↓
Leaf vs Non-Leaf Classification Model
        ↓ (only if leaf)
Plant Disease Classification Model
        ↓
Confidence & Uncertainty Validation
        ↓
Final Prediction or Safe Rejection
```

This design mirrors best practices used in safety-critical computer vision systems such as medical imaging and autonomous driving.

---

## 5. Model 1: Leaf vs Non-Leaf Classification

### 5.1 Purpose

The first model acts as a **gatekeeper** to determine whether the uploaded image belongs to the domain of plant leaf images. Any image that does not contain a plant leaf is rejected before disease classification.

### 5.2 Classes

- **Leaf** – images containing plant leaves (healthy or diseased)
- **Other** – all non-leaf images

### 5.3 Dataset Design

- Leaf images are collected from all plants (Tomato, Potato, Maize) and all disease categories
- Other images include:
  - Furniture posters
  - Buildings
  - Human faces
  - Soil-only images
  - Tools and objects
  - Indoor scenes
  - Screenshots and blurred images

### 5.4 Dataset Size

- Leaf images: ~3000
- Other images: ~800–1000 (20–30% of leaf dataset)

This ratio ensures sufficient diversity without over-biasing the model toward rejection.

### 5.5 Model Characteristics

- Lightweight CNN architecture (e.g., MobileNet or EfficientNet)
- Binary classification
- Fast inference suitable for real-time API usage

---

## 6. Model 2: Leaf Disease Classification

### 6.1 Purpose

The second model is responsible for disease classification and is only executed when the input image has been confirmed to be a valid leaf image.

### 6.2 Plants Covered

- Tomato
- Potato
- Maize

All diseases are leaf-based.

### 6.3 Classes

- Plant-specific disease categories
- Healthy leaf class

This model operates under a **closed-set assumption**, which is acceptable because invalid inputs are already filtered by Model 1.

### 6.4 Dataset

- Only leaf images
- Balanced across plants and disease categories
- Standard train / validation / test split

---

## 7. Training Strategy (Jupyter Notebook)

### 7.1 Data Augmentation

To improve generalization and robustness:
- Random resized cropping
- Rotation and flipping
- Brightness and contrast adjustment
- Normalization

### 7.2 Confidence Calibration

Neural networks are known to produce overconfident probabilities. To address this, **temperature scaling** is applied after training using validation data. This ensures that predicted probabilities better reflect true confidence.

### 7.3 Out-of-Distribution Evaluation

During evaluation, both models are tested using:
- Random internet images
- Non-leaf images
- Poor-quality images

Expected behavior:
- Leaf model classifies them as `Other`
- Disease model is never invoked for such inputs

---

## 8. Deployment Strategy (API Server)

### 8.1 Input Validation

Before inference:
- Image size validation
- RGB format validation
- Blur detection

Invalid images are rejected immediately.

### 8.2 Cascaded Inference Logic

1. Input image is passed to the Leaf vs Non-Leaf model
2. If classified as `Other`, the request is rejected with a safe message
3. If classified as `Leaf`, the image is passed to the Disease model
4. Disease prediction is validated using confidence thresholds

### 8.3 Confidence Thresholding

- Predictions with confidence below a predefined threshold (e.g., 0.80) are rejected
- This prevents uncertain predictions from being shown to users

### 8.4 Uncertainty Handling

Prediction entropy is optionally used to measure uncertainty. High-entropy predictions are treated as unreliable and rejected.

---

## 9. API Response Design

### 9.1 Rejected Input Example

```json
{
  "status": "rejected",
  "reason": "Input image is not a plant leaf"
}
```

### 9.2 Successful Prediction Example

```json
{
  "status": "success",
  "plant": "Tomato",
  "disease": "Early Blight",
  "confidence": 0.91
}
```

This structured response ensures transparency and user trust.

---

## 10. Justification and Industry Alignment

The proposed approach aligns with real-world industry practices:
- Cascaded models reduce false positives
- Explicit rejection of out-of-distribution inputs
- Confidence and uncertainty-based decision-making

Such designs are commonly used in medical imaging, autonomous systems, and agricultural AI platforms where incorrect predictions can lead to serious consequences.

---

## 11. Conclusion

This project adopts a robust, safety-first approach to plant disease detection by combining hierarchical classification, explicit out-of-distribution handling, and API-level validation. The resulting system is not only accurate but also reliable, explainable, and suitable for real-world deployment, meeting the standards expected of a high-quality Final Year Project.

