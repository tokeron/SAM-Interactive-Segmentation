---
title: SAM
app_file: fixed_sam_interface.py
sdk: gradio
sdk_version: 5.44.1
---
# 🎭 SAM 2.0 Interactive Segmentation App

A beautiful and intuitive Streamlit web application for interactive image segmentation using Meta's Segment Anything Model 2.0 (SAM 2.0).

## ✨ Features

- **📷 Image Upload**: Support for JPG, PNG, BMP image formats
- **🎯 Interactive Point Selection**: Click or manually input positive/negative points
- **🖼️ Full-Size Display**: View images in high resolution with point overlays
- **🤖 SAM 2.0 Integration**: Powered by Meta's latest segmentation model
- **📊 Real-Time Visualization**: See segmentation masks overlaid on original images
- **💾 Export Options**: Download masks, overlays, and metadata as PNG/JSON
- **⚙️ Adjustable Parameters**: Control mask threshold and point types

## 🚀 Quick Start

### Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run streamlit_sam_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 How to Use

### 1. Upload an Image
- Click "📷 Upload an image" to select an image file
- Supported formats: JPG, JPEG, PNG, BMP

### 2. Add Points
Choose between **Positive** (include) or **Negative** (exclude) point mode:

#### Quick Presets:
- **🎯 Center**: Add point at image center
- **↖️ Top-Left**: Add point at top-left quarter
- **↗️ Top-Right**: Add point at top-right quarter
- **🎲 Random**: Add random point anywhere

#### Manual Input:
- Enter X,Y coordinates manually
- Points are validated against image boundaries

### 3. Generate Segmentation Mask
- Click "🎯 Generate Segmentation Mask"
- Adjust the mask threshold in the sidebar (0.0-1.0)
- Wait for SAM 2.0 to process (may take 10-30 seconds)

### 4. View Results
- **Original Image with Points**: Shows your input selections
- **Generated Segmentation Mask**: Red overlay on original image
- **Binary Mask Preview**: Black/white mask for download
- **Statistics**: Pixel counts and coverage percentage

### 5. Download Results
- **📥 Download Mask (PNG)**: Binary mask file
- **📥 Download Overlay (PNG)**: Mask overlaid on original
- **📥 Download Data (JSON)**: Complete metadata and statistics

## 🎛️ Advanced Controls

### Sidebar Options:
- **Point Mode**: Switch between Positive/Negative points
- **Mask Threshold**: Control mask sensitivity (lower = larger masks)
- **Clear Points**: Remove all points at once

### Point Management:
- View all current points with coordinates
- Delete individual points with 🗑️ buttons
- Real-time count of positive/negative points

## 🔧 Technical Details

### SAM 2.0 Model
- Uses `facebook/sam2-hiera-small` by default
- Automatically downloads model weights on first run
- Runs on GPU if available, CPU otherwise

### Dependencies
- `streamlit`: Web interface
- `torch`: PyTorch for model inference
- `transformers`: Hugging Face model loading
- `PIL`: Image processing
- `matplotlib`: Visualization
- `numpy`: Numerical operations
- `opencv-python`: Image processing utilities

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- GPU recommended for faster processing

## 🐛 Troubleshooting

### Common Issues:

1. **Model Download Fails**:
   - Check internet connection
   - Ensure Hugging Face access (may require token for some models)

2. **CUDA Out of Memory**:
   - Try smaller model size
   - Reduce image resolution
   - Use CPU mode: set `CUDA_VISIBLE_DEVICES=""`

3. **Slow Processing**:
   - Use GPU if available
   - Try `sam2-hiera-tiny` model for faster inference

4. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

## 📁 File Structure

```
SAM/
├── streamlit_sam_app.py    # Main application
├── fixed_sam_interface.py  # Original Gradio version
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🎨 Interface Screenshots

The app features a clean, modern interface with:
- Full-width image display
- Intuitive sidebar controls
- Real-time point visualization
- Side-by-side result comparison
- Comprehensive download options

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests!

## 📄 License

This project uses Meta's SAM 2.0 model. Please refer to Meta's license terms for the model weights.

## 🙏 Acknowledgments

- Meta AI for the incredible SAM 2.0 model
- Streamlit for the amazing web app framework
- Hugging Face for model hosting
- The open-source community for all the dependencies