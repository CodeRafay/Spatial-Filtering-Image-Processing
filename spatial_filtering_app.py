# smoothing_sharpening_app.py
import io
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# -------------------------
# Utility / Filter Functions
# -------------------------
st.set_page_config(
    page_title="Smoothing & Sharpening Dashboard", layout="wide")


def to_uint8(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    if img.dtype in [np.float32, np.float64]:
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)

# Noise


def add_gaussian_noise(img, sigma=10):
    gauss = np.random.normal(0, sigma/255.0, img.shape)
    noisy = img/255.0 + gauss
    noisy = np.clip(noisy, 0, 1)
    return (noisy*255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.02):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * img.size * 0.5).astype(int)
    # salt
    coords = [np.random.randint(0, i, num_salt) for i in img.shape]
    noisy[tuple(coords)] = 255
    # pepper
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
    noisy[tuple(coords)] = 0
    return noisy

# Smoothing


def mean_filter(img, ksize=3):
    return cv2.blur(img, (ksize, ksize))


def min_filter(img, ksize=3):
    return ndi.minimum_filter(img, size=ksize).astype(np.uint8)


def max_filter(img, ksize=3):
    return ndi.maximum_filter(img, size=ksize).astype(np.uint8)


def median_filter(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def _mode_of_window(window):
    vals, counts = np.unique(window, return_counts=True)
    return vals[np.argmax(counts)]


def mode_filter(img, size=3):
    if size % 2 == 0:
        size += 1
    return ndi.generic_filter(img, function=_mode_of_window, size=(size, size)).astype(np.uint8)

# Sharpening


def sobel_filter(img, ksize=3):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    return np.clip(mag, 0, 255).astype(np.uint8)


def laplacian_filter(img, ksize=3):
    """Return signed Laplacian (not absolute) for proper sharpening."""
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    return lap  # keep signed values


def sobel_then_laplacian(img, sobel_ksize=3, lap_ksize=3):
    s = sobel_filter(img, ksize=sobel_ksize)
    lap = cv2.Laplacian(s, cv2.CV_64F, ksize=lap_ksize)
    return np.clip(np.absolute(lap), 0, 255).astype(np.uint8)

# Combined helper


def apply_smoothing_then_sharpening(img, smoothing_name, smoothing_param, sharpening_name, sharpening_param):
    img_u8 = to_uint8(img)
    # smoothing
    if smoothing_name == "Mean":
        smooth = mean_filter(img_u8, smoothing_param)
    elif smoothing_name == "Median":
        smooth = median_filter(img_u8, smoothing_param)
    elif smoothing_name == "Mode":
        smooth = mode_filter(img_u8, smoothing_param)
    elif smoothing_name == "Min":
        smooth = min_filter(img_u8, smoothing_param)
    elif smoothing_name == "Max":
        smooth = max_filter(img_u8, smoothing_param)
    else:
        smooth = img_u8.copy()

    # sharpening (operate on smooth)
    if sharpening_name == "Sobel":
        sharp = sobel_filter(smooth, sharpening_param)
    elif sharpening_name == "Laplacian":
        lap = laplacian_filter(smooth, sharpening_param)
        sharp = np.clip(smooth.astype(np.float64) -
                        lap, 0, 255).astype(np.uint8)
    elif sharpening_name == "Sobel + Laplacian":
        sharp = sobel_then_laplacian(
            smooth, sharpening_param, sharpening_param)
    else:
        sharp = smooth.copy()

    return smooth, sharp

# Metrics


def compute_psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def compute_ssim(a, b):
    try:
        return ssim(a, b, data_range=255)
    except Exception:
        return None


# -------------------------
# UI
# -------------------------
st.title("Smoothing & Sharpening Filters Dashboard")
st.markdown("""
This app demonstrates smoothing and sharpening filters (Mean, Median, Mode, Min, Max; Sobel, Laplacian, Sobel+Laplacian).
It supports noise injection, filter parameter control, combined pipelines, and metrics (PSNR, SSIM).
""")

with st.sidebar:
    st.header("Controls")

    uploaded = st.file_uploader(
        "Upload image (grayscale preferred)", type=["png", "jpg", "jpeg"])

    st.subheader("Noise (optional)")
    noise_type = st.selectbox(
        "Noise type", ["None", "Gaussian", "Salt & Pepper"])
    gaussian_sigma = st.slider(
        "Gaussian sigma (std dev)", 0.0, 50.0, 10.0, step=0.5)
    sp_amount = st.slider("Salt & Pepper amount", 0.0, 0.5, 0.02, step=0.005)

    st.subheader("Smoothing")
    smoothing_method = st.selectbox(
        "Smoothing method", ["Mean", "Median", "Mode", "Min", "Max"])
    smoothing_ksize = st.slider(
        "Smoothing window / kernel (odd preferred)", 1, 15, 3, step=2)

    st.subheader("Sharpening")
    sharpening_method = st.selectbox(
        "Sharpening method", ["Sobel", "Laplacian", "Sobel + Laplacian"])
    sharpening_ksize = st.slider("Sharpening kernel/ksize", 1, 7, 3, step=2)

    st.subheader("Pipeline")
    apply_individual = st.button("Apply Smoothing Only")
    apply_sharpen = st.button("Apply Sharpening Only")
    apply_combo = st.button("Apply Smoothing then Sharpening")

    st.markdown("---")
    st.markdown("**Download**")
    if "last_result" in st.session_state and st.session_state["last_result"] is not None:
        buf = io.BytesIO()
        Image.fromarray(st.session_state["last_result"]).save(
            buf, format="PNG")
        st.download_button(
            "📥 Download Processed Image",
            data=buf.getvalue(),
            file_name="processed.png",
            mime="image/png",
        )

    st.markdown("---")
    st.caption(
        "ℹ️ All uploaded images are converted to grayscale internally for processing.")

# state holder
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None
if 'original' not in st.session_state:
    st.session_state['original'] = None

# load image
if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Unable to read image. Make sure file is an image.")
    else:
        st.session_state['original'] = to_uint8(img)
else:
    img = None


def show_matplotlib_figure(fig):
    st.pyplot(fig)


# Results area
if st.session_state['original'] is None:
    st.info("Upload an image to begin (or try one of the sample images).")
else:
    original = st.session_state['original']
    # apply optional noise first
    noisy = original.copy()
    if noise_type == "Gaussian":
        noisy = add_gaussian_noise(original, sigma=gaussian_sigma)
    elif noise_type == "Salt & Pepper":
        noisy = add_salt_pepper_noise(original, amount=sp_amount)

        # Reset last result whenever noise type or params change
    if "prev_noise_type" not in st.session_state:
        st.session_state["prev_noise_type"] = noise_type
    if "prev_gaussian_sigma" not in st.session_state:
        st.session_state["prev_gaussian_sigma"] = gaussian_sigma
    if "prev_sp_amount" not in st.session_state:
        st.session_state["prev_sp_amount"] = sp_amount

    if (noise_type != st.session_state["prev_noise_type"] or
        gaussian_sigma != st.session_state["prev_gaussian_sigma"] or
            sp_amount != st.session_state["prev_sp_amount"]):
        st.session_state["last_result"] = None

    st.session_state["prev_noise_type"] = noise_type
    st.session_state["prev_gaussian_sigma"] = gaussian_sigma
    st.session_state["prev_sp_amount"] = sp_amount

    # Buttons actions
    if apply_individual:
        if smoothing_method == "Mean":
            smooth = mean_filter(noisy, smoothing_ksize)
        elif smoothing_method == "Median":
            smooth = median_filter(noisy, smoothing_ksize)
        elif smoothing_method == "Mode":
            smooth = mode_filter(noisy, smoothing_ksize)
        elif smoothing_method == "Min":
            smooth = min_filter(noisy, smoothing_ksize)
        elif smoothing_method == "Max":
            smooth = max_filter(noisy, smoothing_ksize)
        st.session_state['last_result'] = smooth
        st.success(f"{smoothing_method} smoothing applied")

    if apply_sharpen:
        if sharpening_method == "Sobel":
            sharpened = sobel_filter(noisy, sharpening_ksize)
            st.session_state['last_result'] = sharpened
            st.success("Sobel sharpening applied")
        elif sharpening_method == "Laplacian":
            lap = laplacian_filter(noisy, sharpening_ksize)
            sharpened = np.clip(noisy.astype(np.float64) -
                                lap, 0, 255).astype(np.uint8)
            st.session_state['last_result'] = sharpened
            st.success("Laplacian sharpening applied")
        else:
            sl = sobel_then_laplacian(
                noisy, sharpening_ksize, sharpening_ksize)
            st.session_state['last_result'] = sl
            st.success("Sobel + Laplacian applied")

    if apply_combo:
        smooth, sharp = apply_smoothing_then_sharpening(
            noisy,
            smoothing_method,
            smoothing_ksize,
            sharpening_method,
            sharpening_ksize
        )
        st.session_state['last_result'] = sharp
        st.success(
            f"Pipeline applied: {smoothing_method} then {sharpening_method}")

    # Display outputs
    tabs = st.tabs(["Images", "Histograms", "Metrics", "Kernels & Notes"])
    with tabs[0]:
        st.subheader("Image Views")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original, caption="Original",
                     use_container_width=True, clamp=True)
        with col2:
            st.image(noisy, caption="Input (with noise if applied)",
                     use_container_width=True, clamp=True)
        with col3:
            if st.session_state['last_result'] is not None:
                st.image(st.session_state['last_result'],
                         caption="Last Result", use_container_width=True, clamp=True)
            else:
                st.write("No result yet. Use the control buttons.")

       # Laplacian 3-image special view
        if st.session_state['last_result'] is not None and sharpening_method == "Laplacian":
            st.markdown(
                "### Laplacian special view (Smoothed, Laplacian Response, Sharpened)")

            lap = laplacian_filter(noisy, sharpening_ksize)
            lap_display = np.clip(lap, 0, 255).astype(np.uint8)
            diff_img = np.clip(noisy.astype(np.float64) -
                               lap, 0, 255).astype(np.uint8)

            colA, colB, colC = st.columns(3)
            with colA:
                st.image(noisy, caption="Input (after noise)",
                         use_container_width=True, clamp=True)
            with colB:
                st.image(lap_display, caption="Laplacian Response",
                         use_container_width=True, clamp=True)
            with colC:
                st.image(diff_img, caption="Sharpened (Input - Laplacian)",
                         use_container_width=True, clamp=True)

    with tabs[1]:
        st.subheader("Histograms")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(original.ravel(), bins=256, range=(0, 255))
        axes[0].set_title("Original")
        axes[1].hist(noisy.ravel(), bins=256, range=(0, 255))
        axes[1].set_title("Noisy / Input")
        if st.session_state['last_result'] is not None:
            axes[2].hist(st.session_state['last_result'].ravel(),
                         bins=256, range=(0, 255))
            axes[2].set_title("Last Result")
        else:
            axes[2].text(0.5, 0.5, "No result yet",
                         horizontalalignment='center', verticalalignment='center')
            axes[2].set_title("Last Result")
        plt.tight_layout()
        show_matplotlib_figure(fig)

    with tabs[2]:
        st.subheader("Quantitative Metrics")
        if st.session_state['last_result'] is None:
            st.write("No result to evaluate. Apply a filter or pipeline first.")
        else:
            last = to_uint8(st.session_state['last_result'])
            psnr_value = compute_psnr(original, last)
            ssim_value = compute_ssim(original, last)
            st.metric("PSNR (vs original)", f"{psnr_value:.3f}" if np.isfinite(
                psnr_value) else "inf")
            st.metric("SSIM (vs original)",
                      f"{ssim_value:.4f}" if ssim_value is not None else "N/A")
            st.markdown("""
            **Interpretation:** Higher PSNR/SSIM usually indicates the processed image is closer to original.
            For sharpening tasks these values can drop because sharpening intentionally changes local intensities 
            to emphasize edges — use visual inspection alongside metrics.
            """)

    with tabs[3]:
        st.subheader("Filter Kernels & Notes")
        st.markdown("**Sobel (x kernel)**")
        st.latex(
            r"\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}")
        st.markdown("**Sobel (y kernel)**")
        st.latex(
            r"\begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}")
        st.markdown("**Laplacian (example kernel)**")
        st.latex(
            r"\begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}")
        st.markdown(
            "**Mean (k×k)**: each element = 1/k^2. Median and Mode are order-statistics filters (no linear kernel).")
        st.markdown("""
        **Teacher guidelines checklist**:
        - Implemented Sobel (1st derivative) and Laplacian (2nd derivative).
        - Smoothing: Mean, Median, Mode, Min, Max implemented.
        - Noise simulation: Gaussian and Salt & Pepper with adjustable strength.
        - Combined pipeline available.
        - Visual outputs + histograms + kernels + metrics (PSNR, SSIM).
        """)
