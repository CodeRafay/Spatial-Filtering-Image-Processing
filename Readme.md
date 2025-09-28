# Spatial Filtering Dashboard in Image Processing

An interactive Google Colab dashboard to demonstrate **smoothing** and **sharpening** filters for image processing.  
Includes Mean, Median, and Mode smoothing filters, along with Sobel, Laplacian, and Sobel+Laplacian sharpening filters.  
Useful for **noise reduction, edge detection, and detail enhancement** in real-world applications.

---

## 🔍 Introduction

Spatial filtering is an essential step in many computer vision and image processing tasks.  
It helps in:

- Enhancing details
- Detecting edges
- Reducing noise  
  This project demonstrates how different filters behave and how smoothing can be combined with sharpening to improve results.

---

## 📂 Dataset

You can use:

- Any **grayscale or color image** uploaded through the dashboard.
- Recommended: datasets where **edges and details matter**, such as medical images, satellite images, or defect detection samples.

---

## ⚙️ Methodology & Justification

The following filters are implemented:

### **Smoothing Filters**

- **Mean Filter:** Averages local pixels, reduces Gaussian-like noise.
- **Median Filter:** Non-linear filter effective against salt-and-pepper noise.
- **Mode Filter:** Replaces pixels with the most frequent neighbor value, good for impulse noise.

### **Sharpening Filters**

- **Sobel Filter (First Derivative):** Highlights intensity gradients and edges.
- **Laplacian Filter (Second Derivative):** Captures fine details, sensitive to noise.
- **Sobel + Laplacian:** Combines directional edge detection with second-order enhancement.

### **Combined Pipeline**

Smoothing → Sharpening  
Prevents noise amplification, produces clearer edges.

---

## 🚀 Setup

Clone the repo and open in **Google Colab**:

```bash
git clone https://github.com/CodeRafay/Spatial-Filtering-Image-Processing.git
cd spatial-filtering-image-processing
```

Then open the notebook in Colab and run all cells.
Required libraries:

```bash
pip install opencv-python-headless matplotlib ipywidgets scipy
```

---

## 📊 Usage

1. Upload an image from your local machine.
2. Choose a **smoothing filter** (Mean, Median, Mode).
3. Choose a **sharpening filter** (Sobel, Laplacian, Sobel+Laplacian).
4. Optionally enable **"Apply smoothing before sharpening"** for combined results.
5. Compare original vs processed images, histograms, and filter explanations.

---

## 📈 Results & Analysis

- **Mean filter**: Reduces noise but blurs edges.
- **Median filter**: Preserves edges while removing salt-and-pepper noise.
- **Mode filter**: Stabilizes impulse noise but less common in practice.
- **Sobel**: Good for detecting strong directional edges.
- **Laplacian**: Captures fine details but amplifies noise.
- **Combined smoothing + sharpening**: Produces balanced results, especially with Median + Sobel.

---

## 📜 Citation

If you use this project in research or coursework, please cite:

```bibtex
@software{spatial_filtering_dashboard,
  author = Rafay Adeel,
  title = {Spatial Filtering Dashboard in Image Processing},
  year = {2025},
  url = {https://github.com/CodeRafay/Spatial-Filtering-Image-Processing}
}
```

---

## 🏷️ Keywords

Image Processing, Spatial Filters, Smoothing, Sharpening, Sobel, Laplacian, Median Filter, Edge Detection, Colab Dashboard

