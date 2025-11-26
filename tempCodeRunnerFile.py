import streamlit as st
import cv2
import numpy as np

# Try to import metrics, handle error if library is missing
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Filters Visualization", # RENAMED HERE
    page_icon="👁️",
    layout="wide"
)

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0b1120; color: #e2e8f0; }
    h1, h2, h3 { color: #22d3ee !important; font-family: monospace; }
    .info-box {
        padding: 15px; border-left: 5px solid #22d3ee;
        background-color: #132f40; margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def dft_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def idft_transform(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def create_mask(shape, filter_type, d0, w=10):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    if "Lowpass" in filter_type:
        mask[dist_from_center <= d0] = 1
    elif "Highpass" in filter_type:
        mask[dist_from_center > d0] = 1
    elif "Bandpass" in filter_type:
        mask[(dist_from_center >= d0) & (dist_from_center <= d0 + w)] = 1
    elif "Bandreject" in filter_type:
        mask[:, :] = 1
        mask[(dist_from_center >= d0) & (dist_from_center <= d0 + w)] = 0
    return mask

# --- 4. SCENARIO MAPPING ---
SPATIAL_SCENARIOS = {
    "Redact Faces / Privacy (Gaussian Blur)": "Gaussian Blur",
    "Remove Rain / Dead Pixels (Median Blur)": "Median Blur",
    "Enhance Face for AI (Bilateral Filter)": "Bilateral Filter",
    "Est. Background / Motion (Box Filter)": "Box (Average) Filter",
    "Detect Intruder Edges (Laplacian)": "Laplacian",
    "Read Blurry Plates (Unsharp Masking)": "Unsharp Masking"
}

FREQUENCY_SCENARIOS = {
    "Remove Static Hiss (Ideal Lowpass)": "Ideal Lowpass",
    "Analyze Structural Cracks (Ideal Highpass)": "Ideal Highpass",
    "Isolate Patterns (Ideal Bandpass)": "Ideal Bandpass",
    "Remove Electrical Interference (Ideal Bandreject)": "Ideal Bandreject"
}

CONTEXT_INFO = {
    "Gaussian Blur": "Used to blur faces or sensitive data to comply with privacy laws (GDPR).",
    "Median Blur": "Best for removing 'Salt & Pepper' noise (rain/snow) without blurring structure.",
    "Bilateral Filter": "Crucial for Face ID. Smooths grain but locks onto edges (eyes/jaw).",
    "Box (Average) Filter": "Creates heavy blur to estimate background lighting for motion detection.",
    "Laplacian": "Removes color/lighting to show only edges. Used for perimeter breach detection.",
    "Unsharp Masking": "Increases contrast. Used to make blurry text on license plates legible.",
    "Ideal Lowpass": "Removes high-frequency static hiss from analog feeds.",
    "Ideal Highpass": "Isolates fine details like fingerprints, removing lighting shadows.",
    "Ideal Bandpass": "Isolates specific texture frequencies.",
    "Ideal Bandreject": "Removes periodic noise, such as 60Hz electrical hum bars."
}

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Settings")
    domain = st.selectbox("Processing Pipeline", ["Spatial Domain", "Frequency Domain (FFT)"])
    st.markdown("---")
    
    if domain == "Spatial Domain":
        selected_scenario = st.selectbox("Select Real-Life Application:", list(SPATIAL_SCENARIOS.keys()))
        technical_name = SPATIAL_SCENARIOS[selected_scenario]
        
        if technical_name == "Gaussian Blur":
            k_size = st.slider("Blur Strength", 1, 31, 5, step=2)
        elif technical_name == "Median Blur":
            k_size = st.slider("Rain/Snow Removal Strength", 1, 31, 5, step=2)
        elif technical_name == "Box (Average) Filter":
            k_size = st.slider("Kernel Size", 1, 31, 5, step=2)
        elif technical_name == "Bilateral Filter":
            d = st.slider("Diameter", 1, 30, 9)
            sigma_color = st.slider("Smoothness", 10, 150, 75)
        elif technical_name == "Unsharp Masking":
            strength = st.slider("Sharpening Strength", 0.1, 5.0, 1.5)
    
    else: # Frequency Domain
        selected_scenario = st.selectbox("Select Real-Life Application:", list(FREQUENCY_SCENARIOS.keys()))
        technical_name = FREQUENCY_SCENARIOS[selected_scenario]
        d0 = st.slider("Cutoff Radius", 5, 200, 30)
        w = st.slider("Band Width", 5, 100, 20) if "Band" in technical_name else 0

# --- 6. MAIN APP ---
st.title("Filters Visualization") # RENAMED HERE

desc = CONTEXT_INFO.get(technical_name, "Standard processing.")
st.markdown(f"""
<div class="info-box">
    <h4 style="margin:0"> Application: {selected_scenario}</h4>
    <p style="margin:5px 0 0 0; font-family: sans-serif;">{desc}</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)

    # --- PROCESS ---
    out = None
    if domain == "Spatial Domain":
        if technical_name == "Gaussian Blur":
            out = cv2.GaussianBlur(img, (k_size, k_size), 0)
        elif technical_name == "Median Blur":
            out = cv2.medianBlur(img, k_size)
        elif technical_name == "Box (Average) Filter":
            out = cv2.blur(img, (k_size, k_size))
        elif technical_name == "Bilateral Filter":
            out = cv2.bilateralFilter(img, d, sigma_color, 75)
        elif technical_name == "Laplacian":
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            out = cv2.cvtColor(np.uint8(np.absolute(lap)), cv2.COLOR_GRAY2RGB)
        elif technical_name == "Unsharp Masking":
            blur = cv2.GaussianBlur(img, (5,5), 0)
            out = cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)
            
    else: # Frequency
        fshift = dft_transform(gray)
        mask = create_mask(gray.shape, technical_name, d0, w)
        out_gray = idft_transform(fshift * mask)
        out_gray = cv2.normalize(out_gray, None, 0, 255, cv2.NORM_MINMAX)
        out = cv2.cvtColor(np.uint8(out_gray), cv2.COLOR_GRAY2RGB)

    with col2:
        st.subheader("Filtered Result")
        st.image(out, caption=f"Applied: {technical_name}", use_container_width=True)

    # --- ACCURACY & METRICS SECTION ---
    st.markdown("---")
    st.subheader(" Accuracy & Performance Analysis")
    
    if METRICS_AVAILABLE:
        # Prepare images for metrics (Grayscale is standard for these metrics)
        g_in = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if len(out.shape) == 3:
            g_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        else:
            g_out = out

        # 1. PSNR (Peak Signal-to-Noise Ratio)
        # Higher = Image is cleaner/closer to original. Lower = Image is heavily modified.
        val_psnr = psnr(g_in, g_out)
        
        # 2. SSIM (Structural Similarity Index)
        # 1.0 = Identical. 0.0 = Totally different. 
        # For De-noising, you want High SSIM (removed noise but kept structure).
        val_ssim = ssim(g_in, g_out)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("PSNR (Signal Fidelity)", f"{val_psnr:.2f} dB", help="Higher is usually better. Measures how much valid signal remains vs noise.")
        with m2:
            st.metric("SSIM (Structure Match)", f"{val_ssim:.4f}", help="1.0 is a perfect match. If this is too low, the filter destroyed the image details.")
        
        # 3. VISUAL ACCURACY (Difference Map)
        with m3:
            st.markdown("**Change Detection Map**")
            # Calculate absolute difference
            diff = cv2.absdiff(g_in, g_out)
            # Enhance contrast of the difference so it's visible
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            st.image(diff, caption="White pixels = Areas modified by filter", clamp=True)
            
    else:
        st.warning(" Metrics library not found. Install scikit-image to see Accuracy scores.")