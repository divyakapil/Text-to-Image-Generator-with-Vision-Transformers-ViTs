import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("🖼️ Text-to-Image Generator using Stable Diffusion & ViTs")
st.markdown("Enter a text prompt below and generate an image!")

prompt = st.text_input("Enter your prompt:", "A futuristic city at sunset")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)