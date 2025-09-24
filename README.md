Track Anything with Streamlit
📌 Overview

This project extends the Track-Anything
 framework with a Streamlit-based interface, making it easier to experiment with video object tracking and segmentation. By integrating Segment Anything
 into an interactive web app, users can select objects with simple clicks and track them across multiple frames using masks.

The app is designed to make cutting-edge research more accessible, visual, and interactive, while supporting real-world applications such as:

🎬 Video object tracking & segmentation – across shots, scene changes, and complex motion.

🖌 Annotation tools – for creating high-quality segmentation datasets.

🎨 Video inpainting and editing – removing or replacing objects across entire sequences.

🛠 Research & prototyping – quickly testing and visualizing segmentation workflows.

🚀 Features

Streamlit UI for interactive object selection and tracking.

Mask-based tracking across video frames.

Dynamic corrections – refine or change tracked objects mid-sequence.

Support for multi-object tracking in the same video.

Built on top of Segment Anything + Track-Anything core.

🎥 Example Workflow

Upload or stream a video.

Click on an object of interest in the first frame.

Generate a segmentation mask with Segment Anything.

Track the mask across multiple frames.

Optionally refine, edit, or export the results.

⚡ Installation

Clone the repository and install dependencies:

git clone https://github.com/<your-username>/track-anything-streamlit.git
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
