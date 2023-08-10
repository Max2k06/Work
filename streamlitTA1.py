import random
import streamlit as st
import pandas as pd
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile
from controllers import get_end_number, get_frames_from_video, select_template, vos_tracking_video
from controllers import sam_refine
import skimage
import plotly.express as px
from skimage import io
from track_anything import parse_augment
from model import get_model
from streamlit_plotly_events import plotly_events
from datetime import datetime



# args, defined in track_anything.py
if 'args' not in st.session_state:
    st.session_state['args']= parse_augment()

# initialize sam, xmem, e2fgvi models
if 'model' not in st.session_state:
    st.session_state['model']= get_model(st.session_state['args'])

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

st.session_state['point_prompt']= str(st.radio(
    "Choose point type",["Positive", "Negative"]))

st.write('Note: If mask needs to be removed/exlcuded, place negative point on the mask')

if 'stvideo_state' not in st.session_state:
    st.session_state['stvideo_state']={}

st.button(
    "Extract Frames", on_click= get_frames_from_video, args=(st.session_state['model'], uploaded_video, st.session_state['stvideo_state'])
    )
def delete_fig():
    if 'fig' in st.session_state:
        del st.session_state["fig"]

if 'video_info' in st.session_state['stvideo_state'] :
    if "origin_images" in st.session_state["stvideo_state"]:
        slider_info= st.slider('Choose Frame', min_value=0, max_value=len(st.session_state['stvideo_state']['origin_images'])-1, value=(0, len(st.session_state['stvideo_state']['origin_images'])-1), on_change= delete_fig)
        get_end_number(slider_info[1], st.session_state['stvideo_state'], st.session_state['interactive_state'])
        select_template(st.session_state['model'], slider_info[0], st.session_state['stvideo_state'], st.session_state['interactive_state'])
    st.write(st.session_state['stvideo_state']["video_info"])
    if 'fig' not in st.session_state:
        st.session_state['fig']= px.imshow(st.session_state['stvideo_state']["origin_images"][st.session_state["stvideo_state"]["select_frame_number"]])
    try:
        st.session_state['selected_point']= plotly_events(st.session_state['fig']).pop()
        st.write(st.session_state['selected_point'])
    except IndexError:
        pass
   
if 'interactive_state' not in st.session_state:
    st.session_state['interactive_state']= {"positive_click_times":0, "negative_click_times":0, "inference_times":0, "mask_save":True}

if 'click_state' not in st.session_state:
    st.session_state['click_state']= [[], []]

if 'selected_point' in st.session_state:
    x_coordinate= st.session_state['selected_point']['x']
    y_coordinate= st.session_state['selected_point']['y']
    evt= (x_coordinate, y_coordinate)
    painted_image, video_state, interactive_state, operation_log= sam_refine(st.session_state['model'], st.session_state['stvideo_state'], st.session_state['point_prompt'],  st.session_state['interactive_state'], evt, st.session_state['click_state'])
    st.write(st.session_state['click_state'])
    st.session_state['fig'] = px.imshow(painted_image)
    del st.session_state['selected_point']
    st.experimental_rerun()

if 'video_info' in st.session_state['stvideo_state']:
    st.button("Track", on_click=vos_tracking_video,args=(st.session_state['model'], st.session_state['stvideo_state'], st.session_state['interactive_state']))

if 'video_output' in st.session_state['stvideo_state']:
    print(st.session_state['stvideo_state']['video_output'])
    st.video(st.session_state['stvideo_state']['video_output'])