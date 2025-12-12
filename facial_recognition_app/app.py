import cv2
import numpy as np
from PIL import Image
import streamlit as st

# -------------------------
# Load the DNN face model
# -------------------------
@st.cache_resource
def load_face_net():
    prototxt = "model/deploy.prototxt"
    weights = "model/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, weights)
    return net

def detect_and_annotate(image_rgb, mode="Boxes", conf_threshold=0.5):
    net = load_face_net()

    # Convert to OpenCV BGR and get dimensions
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    (h, w) = image_bgr.shape[:2]

    # Prepare blob for DNN
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 117.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    detections = net.forward()

    output = image_bgr.copy()
    faces = []

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue

        # Scale box back up to original image size
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        faces.append((x1, y1, x2, y2))

        if mode == "Boxes":
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        else:
            face_roi = output[y1:y2, x1:x2]

            if mode == "Blur faces":
                face_roi = cv2.GaussianBlur(face_roi, (0, 0), sigmaX=15, sigmaY=15)

            elif mode == "Pixelate faces":
                h_roi, w_roi = face_roi.shape[:2]
                small = cv2.resize(face_roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                face_roi = cv2.resize(small, (w_roi, h_roi),
                                      interpolation=cv2.INTER_NEAREST)

            output[y1:y2, x1:x2] = face_roi

    # Convert back to RGB for display
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output_rgb, faces


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Facial Recognition Demo", layout="centered")

st.title("🧠 Facial Recognition with DNN")
st.write(
    """
    This web app uses the **Res10 SSD Caffe model** to detect faces.

    - Upload a photo or use your webcam  
    - Choose to draw boxes, blur faces, or pixelate faces  
    """
)

mode = st.radio(
    "What do you want to do with detected faces?",
    ["Boxes", "Blur faces", "Pixelate faces"],
    horizontal=True,
)

conf_threshold = st.slider(
    "Detection confidence threshold", 0.1, 0.9, 0.5, 0.05
)

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
camera = st.camera_input("Or take a picture with your webcam")

img = None
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
elif camera is not None:
    img = Image.open(camera).convert("RGB")

if img is not None:
    st.image(img, caption="Original image", use_container_width=True)

    if st.button("Run facial recognition"):
        with st.spinner("Running the face detector..."):
            img_np = np.array(img)
            result, faces = detect_and_annotate(
                img_np, mode=mode, conf_threshold=conf_threshold
            )

        st.subheader("Result")
        st.image(result, use_container_width=True)
        st.success(f"Detected {len(faces)} face(s).")
else:
    st.info("Upload an image or use the camera to get started.")
