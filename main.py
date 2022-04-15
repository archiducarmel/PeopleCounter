import streamlit as st
from PIL import Image
import cv2
import numpy as np
from yolov5.helpers import load_model
import yolov5
import tempfile
from pathlib import Path


def load_dlmodel(model_file, device):
    model = load_model(model_path=model_file, device=device)
    return model

st.title("Comptage de Personnes")
img = Image.open("giroud.png")
st.image(img, width=100)

img = Image.open("./line.jpg")
st.image(img, width=750)

##### CHARGEMENT DE L'IMAGE #################
uploaded_image = st.sidebar.file_uploader("Charger votre image", type=['jpg', "png", "tiff", "jpeg"], key=3)

if uploaded_image is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_image_file1:
        fp1 = Path(tmp_image_file1.name)
        fp1.write_bytes(uploaded_image.getvalue())
        frame = cv2.imread(tmp_image_file1.name)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

if uploaded_image is not None:
    image_is_valid = True
else:
    image_is_valid = False

model_path = "./yolov5l.pt"

if image_is_valid:
    model = load_dlmodel(model_file=model_path, device="cpu")

    img = Image.fromarray(frame)

    results = model(img, augment=True)
    bboxes = results.pred[0].tolist()
    human_bboxes = [box[:4] for box in bboxes if int(box[5]) == 0]

    img_results = results.render()

    st.image(img_results, width=500)

    col1, col2, col3, col4 = st.columns(4)

    col2.write("Personnes détectées :")
    col3.write(len(human_bboxes))





