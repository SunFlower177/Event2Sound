# Event2Sound

An event-based audio reconstruction system that recovers sound from event camera data by extracting micro-vibration signals using a Riesz pyramid and reconstructing audio via frequency-band filtering and PCA.

---

## 📌 Overview

**Event2Sound** is a research-oriented project that explores how **sound can be recovered from event camera data**.

Instead of relying on traditional frame-based cameras, this system uses an **event-based vision sensor** to capture micro-vibrations caused by sound waves. These vibrations are then analyzed in the **phase domain** using a **Riesz pyramid**, filtered in relevant frequency bands, and finally reconstructed into an audible waveform using **PCA-based signal reconstruction**.

This project demonstrates how event-based vision and signal processing can be combined to recover audio signals from purely visual event streams.

---

## ✨ Key Features

- 🎧 **Audio reconstruction from event camera data**
- ⚡ **Event-based processing (no RGB frames required)**
- 🧠 **Riesz pyramid phase analysis** for micro-vibration extraction
- 🎚️ **Frequency band–aware signal selection**
- 📉 **Memory-safe phase unwrapping and filtering**
- 📊 **PCA-based denoising and signal reconstruction**
- 🎼 **Waveform and spectrogram visualization**

---

## 🧠 Method Overview

The overall pipeline consists of the following steps:

```text
Event Camera RAW Data
        ↓
Event Stream Aggregation (Signed Pseudo-Frames)
        ↓
Laplacian Pyramid Construction
        ↓
Riesz Transform (Amplitude & Phase Extraction)
        ↓
Active Pixel Selection (Amplitude-based)
        ↓
Phase Unwrapping (Memory-safe)
        ↓
Band-pass Filtering (Target Frequency Range)
        ↓
Robust Standardization
        ↓
PCA-based Audio Reconstruction
        ↓
Waveform & Spectrogram Output
```
---

## 🛠️ Implementation Details

## 1.Riesz Pyramid Analysis ##

Builds a Laplacian pyramid for multi-scale analysis

Applies the Riesz transform at each pyramid level

Extracts local amplitude and phase information

Designed to preserve phase precision for audio reconstruction

 ## 2.Signal Extraction & Processing ##

Selects active pixels based on mean amplitude

Avoids full-frame phase unwrapping to prevent memory explosion

Applies band-pass filtering to isolate vibration frequencies

Uses robust standardization (median & MAD) before PCA

## 3.Audio Reconstruction ##

Uses PCA to extract the dominant vibration component

Applies post-processing:(1)Detrending;(2)Band-pass filtering;(3)Amplitude normalization

Outputs: .wav audio file & Waveform & spectrogram visualization

---

## 🎯 Applications

Event-based audio sensing

Micro-vibration analysis

Non-contact sound recovery

Event-based signal processing research

Computational photography & vision research

---

## 🔮 Future Work

Multi-source audio separation

Deep learning–based reconstruction

Adaptive frequency band selection

Real-time processing optimization

Quantitative evaluation with ground-truth audio

---

## 📜 Disclaimer

This project is intended for research and educational purposes only.
It is not designed for surveillance or privacy-intrusive applications.

---
