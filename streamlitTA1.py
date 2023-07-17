import streamlit as st
import pandas as pd
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile
from controllers import get_frames_from_video


st.title('Track Anything')
st.caption('Demo')
st.write('This is a project which makes use of masks in order to track a specific element through numerous frames')

col1, col2= st.columns([2, 1])

uploaded_video= col1.file_uploader("Upload a video to track within")

# Learn how to extract video information(not priority)


#Checklist(Only priorities nothing more, that is later)
#   -Use "st.selectbox" to create the dropdown menu for selecting between masks
#   -Use st.radio, same as gradio, to create the options for negative and positive

st.video(uploaded_video)

mask_selecter= st.selectbox(
    "Choose mask",
    ["mask", "flask", "dusk"]
)
point_selecter= st.radio(
    "Choose point type",
    ["Positive", "Negative"]
)
add_mask_button= st.button(
    "Add Mask"
)
remove_mask_button= st.button(
    "Remove Mask"
)
# st.select_slider(
#     "Select a frame"
#     [for i in range(5)]
# )

# Get frames button to be added
# Wire in the extract frames function to the button , understand where video state and video info are being used, display the content of video info, pass the frames to the segment anything model, acquire the masks from the model
# Display 1st frame and display video info

# extract_frames_button= st.button(
#     "Extract Frames", on_click= get_frames_from_video(video_state=0, uploaded_video)

# )
# if st.button("Extract frames"):
#     st.header("starting")
#     get_frames_from_video
