# Track Anything with Streamlit
# 📌 Overview

This project extends the Track-Anything
 framework with a Streamlit-based interface, making it easier to experiment with video object tracking and segmentation. By integrating Segment Anything
 into an interactive web app, users can select objects with simple clicks and track them across multiple frames using masks.

The app is designed to make cutting-edge research more accessible, visual, and interactive, while supporting real-world applications such as:

<ul>
<li>🎬 Video object tracking & segmentation – across shots, scene changes, and complex motion.</li> 

<li>🖌 Annotation tools – for creating high-quality segmentation datasets.</li>

<li>🎨 Video inpainting and editing – removing or replacing objects across entire sequences.</li>

<li>🛠 Research & prototyping – quickly testing and visualizing segmentation workflows.</li>
</ul>


# 🚀 Features
<ul>
<li>Streamlit UI for interactive object selection and tracking.</li>

<li>Mask-based tracking across video frames.</li>

<li>Dynamic corrections – refine or change tracked objects mid-sequence.</li>

<li>Support for multi-object tracking in the same video.</li>

<li>Built on top of Segment Anything + Track-Anything core.</li>
</ul>

# 🎥 Example Workflow

1. Upload or stream a video.

2. Click on an object of interest in the first frame.

3. Generate a segmentation mask with Segment Anything.

4. Track the mask across multiple frames.

Optionally refine, edit, or export the results.

# ⚡ Installation

Clone the repository and install dependencies:

git clone https://github.com/Max2k06/track-anything-ai-model.git

cd track-anything-streamlit

# Create a virtual environment (recommended)
conda create -n track-anything python=3.9

conda activate track-anything

# Install requirements
pip install -r requirements.txt

▶️ Running the App
streamlit run app.py


Open the provided local URL in your browser to interact with the app.

📚 Citation

If you build on this project, please give credit to the original authors of Track-Anything:

@misc{yang2023track,
      title={Track Anything: Segment Anything Meets Videos}, 
      author={Jinyu Yang and Mingqi Gao and Zhe Li and Shang Gao and Fangjing Wang and Feng Zheng},
      year={2023},
      eprint={2304.11968},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

🙏 Acknowledgements

Track-Anything

Segment Anything

Streamlit
