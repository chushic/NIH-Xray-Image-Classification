import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Load your model
model = torch.load("models/vit_base_1e-05_05.pth")['model']

transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# A function to process the uploaded image
def predict_xray(image, 
                 model_path="models/vit_base_1e-05_05.pth", 
                 device="cpu"):
    transform_from_PIL = transforms.Compose([transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformed = transform_from_PIL(image)
    transformed.to(device)
    model = torch.load(model_path)["model"]
    model.to(device)
    res_dict = ['Atelectasis',
                 'Cardiomegaly',
                 'Consolidation',
                 'Edema',
                 'Effusion',
                 'Emphysema',
                 'Fibrosis',
                 'Hernia',
                 'Infiltration',
                 'Mass',
                 'No Finding',
                 'Nodule',
                 'Pleural_Thickening',
                 'Pneumonia',
                 'Pneumothorax']
    prediction = model(transformed.unsqueeze(0))
    label = res_dict[torch.argmax(prediction)]
    return label

# Streamlit code to create the web app interface
st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict_xray(image) # make_prediction(image)
    st.write(f"The predicted label is: {label}")
