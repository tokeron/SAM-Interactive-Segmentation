#!/usr/bin/env python3
"""
SAM 2.1 Interface
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gradio as gr
from transformers import Sam2Model, Sam2Processor
import warnings
import io
import base64
import os
from datetime import datetime
# Grounding DINO will be imported dynamically in the initialization function

warnings.filterwarnings("ignore")

# Global model instance to avoid reloading
MODEL = None
PROCESSOR = None
DEVICE = None

# Global Grounding DINO instance
GROUNDING_DINO = None

# Global state for saving
CURRENT_MASK = None
CURRENT_IMAGE_NAME = None
CURRENT_POINTS = None

def initialize_sam(model_size="small"):
    """Initialize SAM model once"""
    global MODEL, PROCESSOR, DEVICE
    
    if MODEL is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing SAM 2.1 {model_size} on {DEVICE}...")
        
        model_name = f"facebook/sam2-hiera-{model_size}"
        MODEL = Sam2Model.from_pretrained(model_name).to(DEVICE)
        PROCESSOR = Sam2Processor.from_pretrained(model_name)
        
        print("✓ Model loaded successfully!")
    
    return MODEL, PROCESSOR, DEVICE

def initialize_grounding_dino():
    """Initialize Grounding DINO model once"""
    global GROUNDING_DINO, DEVICE
    
    if GROUNDING_DINO is None:
        if DEVICE is None:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing Grounding DINO on {DEVICE}...")
        
        try:
            # Use Hugging Face model for Grounding DINO
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-RESEARCH/grounding-dino-base"
            GROUNDING_DINO = {
                'processor': AutoProcessor.from_pretrained(model_id),
                'model': AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
            }
            print("✓ Grounding DINO loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load Grounding DINO: {e}")
            print("Note: Falling back to manual point selection only")
            GROUNDING_DINO = None
    
    return GROUNDING_DINO

def detect_objects_with_text(image, text_prompt, confidence_threshold=0.25):
    """Use Grounding DINO to detect objects based on text prompt"""
    global GROUNDING_DINO
    
    try:
        # Initialize Grounding DINO if needed
        grounding_dino = initialize_grounding_dino()
        if grounding_dino is None:
            return None, "❌ Grounding DINO not available"
        
        # Fix image format
        pil_image = fix_image_array(image)
        
        # Prepare inputs for Grounding DINO
        processor = grounding_dino['processor']
        model = grounding_dino['model']
        
        # Process inputs
        inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process results
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=confidence_threshold,
            text_threshold=0.25,
            target_sizes=[pil_image.size[::-1]]  # (height, width)
        )[0]
        
        if len(results['boxes']) == 0:
            return None, f"No objects found for prompt: '{text_prompt}'"
        
        # Convert boxes to the format expected by SAM [x1, y1, x2, y2]
        detected_boxes = []
        for box in results['boxes']:
            x1, y1, x2, y2 = box.tolist()
            detected_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return detected_boxes, f"✓ Found {len(detected_boxes)} object(s) for '{text_prompt}'"
        
    except Exception as e:
        return None, f"❌ Detection failed: {str(e)}"

def fix_image_array(image):
    """Fix image input for SAM processing - handles filepath, numpy array, or PIL Image"""
    if isinstance(image, str):
        # Handle filepath input from Gradio
        return Image.open(image).convert("RGB")
    
    elif isinstance(image, np.ndarray):
        # Make sure array is contiguous
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        # Ensure uint8 dtype
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to PIL Image to avoid any stride issues
        return Image.fromarray(image).convert("RGB")
    
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def apply_mask_post_processing(mask, stability_threshold=0.95):
    """Apply post-processing to refine mask size and quality"""
    import cv2
    
    # Convert to binary mask
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Apply morphological operations to clean up the mask
    kernel_size = max(3, int(mask.shape[0] * 0.01))  # Adaptive kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close small holes
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return binary_mask.astype(np.float32)

def apply_erosion_dilation(mask, erosion_dilation_value):
    """Apply erosion or dilation to adjust mask size"""
    import cv2
    
    binary_mask = (mask > 0).astype(np.uint8)
    
    if erosion_dilation_value == 0:
        return mask
    
    kernel_size = abs(erosion_dilation_value)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if erosion_dilation_value > 0:
        # Dilate (make larger)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    else:
        # Erode (make smaller)
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    
    return binary_mask.astype(np.float32)

def save_binary_mask(mask, image_name, points, mask_threshold, erosion_dilation, save_low_res=False, custom_folder_name=None):
    """Save binary mask to organized folder structure"""
    global CURRENT_MASK, CURRENT_IMAGE_NAME, CURRENT_POINTS
    
    try:
        # Store current state for saving
        CURRENT_MASK = mask
        CURRENT_IMAGE_NAME = image_name
        CURRENT_POINTS = points
        
        # Extract image name without extension and sanitize
        if image_name:
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            # Remove any path separators and special characters
            base_name = base_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
        else:
            base_name = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Choose folder tag: user-provided name if available, else 'default'
        folder_tag = None
        if custom_folder_name and str(custom_folder_name).strip():
            folder_tag = str(custom_folder_name).strip().replace(' ', '_')
        else:
            folder_tag = "default"
        


        # Create folder structure: masks/<image_base>/<folder_tag>/
        folder_name = f"masks/{base_name}/{folder_tag}"
        os.makedirs(folder_name, exist_ok=True)
        
        # Create binary mask (0 and 255 values)
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # Calculate low resolution dimensions if requested
        original_height, original_width = binary_mask.shape
        if save_low_res:
            # Calculate sqrt-based resolution
            sqrt_factor = int(np.sqrt(max(original_width, original_height)))
            low_res_width = sqrt_factor
            low_res_height = sqrt_factor
            print(f"Original mask size: {original_width}x{original_height}")
            print(f"Low-res mask size: {low_res_width}x{low_res_height}")
        
        # Save binary mask
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize filename - replace problematic characters
        threshold_str = f"{mask_threshold:.2f}".replace('.', 'p')  # 0.30 -> 0p30
        adj_str = f"{erosion_dilation:+d}".replace('+', 'plus').replace('-', 'minus')  # +2 -> plus2, -2 -> minus2
        
        saved_paths = []
        
        # Save full resolution mask as JPEG with a simple filename
        mask_filename = "image.jpg"
        mask_path = os.path.join(folder_name, mask_filename)
        
        mask_image = Image.fromarray(binary_mask, mode='L')
        mask_image.save(mask_path, format="JPEG", quality=95, optimize=True)
        saved_paths.append(mask_path)

        # Save tensor mask (.pt) as float tensor (0.0/1.0)
        tensor_filename = "image.pt"
        tensor_path = os.path.join(folder_name, tensor_filename)
        torch.save(torch.from_numpy((mask > 0).astype(np.float32)), tensor_path)
        saved_paths.append(tensor_path)
        
        # Save low resolution mask if requested
        if save_low_res:
            # Resize mask to low resolution
            low_res_mask = mask_image.resize((low_res_width, low_res_height), Image.Resampling.NEAREST)
            
            low_res_filename = f"mask_lowres_{sqrt_factor}x{sqrt_factor}_t{threshold_str}_adj{adj_str}_{timestamp}.png"
            low_res_path = os.path.join(folder_name, low_res_filename)
            
            low_res_mask.save(low_res_path)
            saved_paths.append(low_res_path)
        
        # Also save metadata
        metadata = {
            "timestamp": timestamp,
            "points": points,
            "mask_threshold": mask_threshold,
            "erosion_dilation": erosion_dilation,
            "image_name": image_name,
            "original_resolution": f"{original_width}x{original_height}",
            "saved_paths": saved_paths,
            "low_resolution_saved": save_low_res
        }
        
        if save_low_res:
            metadata["low_resolution"] = f"{low_res_width}x{low_res_height}"
            metadata["sqrt_factor"] = sqrt_factor
        
        import json
        metadata_path = os.path.join(folder_name, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Return appropriate message
        if save_low_res:
            return f"✅ Masks saved:\n📁 Full: {os.path.basename(mask_path)}\n📁 Low-res: {os.path.basename(low_res_path)}"
        else:
            return f"✅ Mask saved to: {os.path.basename(mask_path)}"
        
    except Exception as e:
        return f"❌ Save failed: {str(e)}"

def process_sam_segmentation(image, points_data, bbox_data, mode, image_name=None, top_k=3, mask_threshold=0.0, stability_score_threshold=0.95, erosion_dilation=0, text_prompt=None, confidence_threshold=0.25):
    """Main processing function with mask size controls - supports points, bounding boxes, and text prompts"""
    global CURRENT_MASK, CURRENT_IMAGE_NAME, CURRENT_POINTS
    
    if image is None:
        return None, None, "Please upload an image first."
    
    # Check input based on mode
    if mode == "Points":
        if not points_data or len(points_data) == 0:
            return None, None, "Please click on the image to select points."
    elif mode == "Bounding Box":
        if bbox_data is None:
            return None, None, "Please click two corners to define a bounding box."
    elif mode == "Text Prompt":
        if not text_prompt or not text_prompt.strip():
            return None, None, "Please enter a text prompt to detect objects."
    
    try:
        # Initialize model
        model, processor, device = initialize_sam()
        
        # Fix image
        pil_image = fix_image_array(image)
        
        # Prepare SAM inputs based on mode
        input_points = None
        input_labels = None
        input_boxes = None
        points = None
        
        if mode == "Points":
            # Extract points with positive/negative labels
            points = []
            labels = []
            for point_info in points_data:
                if isinstance(point_info, dict):
                    points.append([point_info.get("x", 0), point_info.get("y", 0)])
                    labels.append(1 if point_info.get("positive", True) else 0)  # 1 = positive, 0 = negative
                elif isinstance(point_info, (list, tuple)) and len(point_info) >= 2:
                    points.append([point_info[0], point_info[1]])
                    labels.append(1)  # Default to positive for old format
            
            if not points:
                return None, "No valid points found."
            
            print(f"Processing {len(points)} points: {points} with labels: {labels}")
            input_points = [[points]]
            input_labels = [[labels]]
            
        elif mode == "Bounding Box":
            # Use bounding box
            bbox = bbox_data  # [x1, y1, x2, y2]
            print(f"Processing bounding box: {bbox}")
            input_boxes = [[bbox]]
            # For visualization, store the bbox corners as points
            points = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
            
        elif mode == "Text Prompt":
            # Use Grounding DINO to detect objects from text prompt
            detected_boxes, detection_status = detect_objects_with_text(pil_image, text_prompt, confidence_threshold)
            if detected_boxes is None:
                return None, None, detection_status
            
            # Use the first detected bounding box (highest confidence)
            bbox = detected_boxes[0]
            print(f"Using detected bounding box: {bbox}")
            input_boxes = [[bbox]]
            # For visualization, store the bbox corners as points
            points = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
        
        # Process with SAM
        processor_inputs = {
            "images": pil_image,
            "return_tensors": "pt"
        }
        
        # Add points and/or boxes based on what's available
        if input_points is not None:
            processor_inputs["input_points"] = input_points
            processor_inputs["input_labels"] = input_labels
        
        if input_boxes is not None:
            processor_inputs["input_boxes"] = input_boxes
        
        inputs = processor(**processor_inputs).to(device)
        
        # Generate masks with multiple outputs for better control
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=True)
        
        # Get masks and scores
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]
        
        scores = outputs.iou_scores.cpu().numpy().flatten()
        
        # Get top-k masks and process all of them
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        processed_masks = []
        mask_scores = []
        
        for i, idx in enumerate(top_indices):
            mask = masks[0, idx].numpy()
            score = scores[idx]
            
            # Apply threshold to control mask size
            if mask_threshold > 0:
                mask = (mask > mask_threshold).astype(np.float32)
            
            # Additional mask processing for size control
            mask = apply_mask_post_processing(mask, stability_score_threshold)
            
            # Apply erosion/dilation for fine size control
            if erosion_dilation != 0:
                mask = apply_erosion_dilation(mask, erosion_dilation)
            
            processed_masks.append(mask)
            mask_scores.append(score)
        
        # Store current state for saving (use first mask as default)
        CURRENT_MASK = processed_masks[0]
        CURRENT_IMAGE_NAME = image_name
        CURRENT_POINTS = points
        
        # Create visualizations for the first mask
        original_with_input = create_original_with_input_visualization(pil_image, points, bbox_data, mode)
        mask_result = create_mask_visualization(pil_image, processed_masks[0], mask_scores[0], mask_threshold)
        
        status = f"✓ Generated {len(processed_masks)} masks\n🔄 Use navigation to browse masks"
        
        # Return multiple masks and related data
        return original_with_input, mask_result, status, processed_masks, mask_scores
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return None, None, f"Error: {str(e)}"

def create_original_with_input_visualization(pil_image, points, bbox, mode, negative_points=None):
    """Create visualization of original image with input points/bbox overlay"""
    # Convert PIL to numpy for matplotlib
    img_array = np.array(pil_image)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Show original image only
    ax.imshow(img_array)
    
    # Show input visualization based on mode
    if mode == "Points":
        total_points = 0
        # Show positive points (green)
        if points:
            for point in points:
                ax.plot(point[0], point[1], 'go', markersize=12, markeredgewidth=3, markerfacecolor='lime')
            total_points += len(points)
        
        # Show negative points (red)
        if negative_points:
            for point in negative_points:
                ax.plot(point[0], point[1], 'ro', markersize=12, markeredgewidth=3, markerfacecolor='red')
            total_points += len(negative_points)
            
        pos_count = len(points) if points else 0
        neg_count = len(negative_points) if negative_points else 0
        title_suffix = f"Points: {pos_count}+ {neg_count}-" if neg_count > 0 else f"Points: {pos_count}"
    elif mode == "Bounding Box" and bbox:
        # Show bounding box
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Draw bounding box rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), width, height, linewidth=3, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
        # Show corner points
        ax.plot([x1, x2], [y1, y2], 'go', markersize=8, markeredgewidth=2, markerfacecolor='lime')
        title_suffix = f"BBox: {int(width)}×{int(height)}"
    else:
        title_suffix = "No input"
    
    ax.set_title(f"Input Selection ({title_suffix})", fontsize=14)
    ax.axis('off')
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    result_array = np.asarray(buf)
    # Convert RGBA to RGB
    result_array = result_array[:, :, :3]
    
    plt.close(fig)
    return result_array

def create_mask_visualization(pil_image, mask, score, mask_threshold=0.0):
    """Create clean mask visualization without input overlays"""
    # Convert PIL to numpy for matplotlib
    img_array = np.array(pil_image)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Show original image
    ax.imshow(img_array)
    
    # Overlay mask in red
    mask_overlay = np.zeros((*mask.shape, 4))
    mask_overlay[mask > 0] = [1, 0, 0, 0.6]  # Red with transparency
    ax.imshow(mask_overlay)
    
    ax.set_title(f"Generated Mask (Score: {float(score):.3f}, Threshold: {mask_threshold:.2f})", fontsize=14)
    ax.axis('off')
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    result_array = np.asarray(buf)
    # Convert RGBA to RGB
    result_array = result_array[:, :, :3]
    
    plt.close(fig)
    return result_array

def create_interface():
    """Create a simplified single-image annotator interface."""
    
    with gr.Blocks(title="SAM 2.1 - Simple Annotator", theme=gr.themes.Soft(), css="""
        .negative-mode-checkbox label {
            color: #d00000 !important;
            font-weight: 800 !important;
            font-size: 16px !important;
        }
        """) as interface:
        gr.HTML("""
        <div style="text-align: center;">
            <h1>🎯 AI-Powered Image Segmentation</h1>
            <h2>SAM 2.1 + Grounding DINO</h2>
            <p><strong>✨ Just type what you want to segment!</strong> Try "person", "face", "car", "dog" - or click points manually.</p>
            <p>🎭 Generate multiple mask options and pick your favorite!</p>
            <hr style="margin: 20px 0;">
            <p style="font-size: 12px; color: #666;">
                <strong>Acknowledgment:</strong> This is a GUI interface for research by Meta AI (SAM 2.1) and IDEA Research (Grounding DINO).<br>
                All credit goes to the original researchers. This tool only provides an easy-to-use web interface.
            </p>
        </div>
        """)

        # Image input (single image) - directly annotate; this serves as uploader too
        # Users can upload by clicking the annotatable image component below.
        image_input = gr.Image(
            label=None,
            type="filepath",
            height=0,
            visible=False
        )

        # Text prompt input with clear button
        with gr.Row():
            text_prompt_input = gr.Textbox(
                label="🔍 Text Prompt (Optional)",
                placeholder="Type what to segment (e.g., 'person', 'car', 'dog') and press Enter",
                value="snoopy",
                interactive=True,
                info="💡 Text = auto-detection | Empty + clicking = manual points | Text takes priority if both provided",
                scale=4
            )
            clear_text_btn = gr.Button("🗑️ Clear Text", variant="secondary", scale=1)
        
        # Number of masks to generate
        num_masks = gr.Slider(
            minimum=1,
            maximum=5,
            value=3,
            step=1,
            label="🎭 Number of Masks to Generate",
            info="Generate multiple mask options to choose from"
        )
        
        # Main layout: Selected Points on the left, annotatable image in the center, preview on the right
        with gr.Row():
            with gr.Column(scale=1):
                clear_points_btn = gr.Button("🗑️ Clear Points", variant="secondary", size="sm")
                points_display = gr.JSON(label="📍 Selected Points", value=[], visible=True)
            with gr.Column(scale=3):
                # Negative mode toggle with clear red styling
                negative_point_mode = gr.Checkbox(
                    label="➖ NEGATIVE POINT MODE",
                    value=False,
                    info="🔴 Enable to add negative points (shown in red)",
                    interactive=True,
                    elem_classes="negative-mode-checkbox"
                )
                original_with_input = gr.Image(
                    label="📍 Click to Annotate (toggle negative mode to exclude)",
                    height=640,
                    interactive=True,
                    value="data/snoopy.jpg"
                )
            with gr.Column(scale=1):
                points_overlay = gr.Image(label="📍 Points Preview (green=positive, red=negative)", height=720, interactive=False)

        # Action buttons
        with gr.Row():
            generate_btn = gr.Button("🎯 Generate Mask", variant="primary", size="lg")

        # Mask result with navigation
        with gr.Row():
            mask_result = gr.Image(label="🎭 Generated Mask", height=512)
        
        # Mask navigation controls
        with gr.Row():
            prev_mask_btn = gr.Button("⬅️ Previous", variant="secondary", size="sm")
            mask_info = gr.Textbox(
                label="Mask Info",
                value="No masks generated yet",
                interactive=False,
                scale=2
            )
            next_mask_btn = gr.Button("➡️ Next", variant="secondary", size="sm")

        # Save controls under mask
        with gr.Row():
            mask_name_input = gr.Textbox(label="Folder name (optional)", placeholder="e.g., michael_phelps_bottom_left", scale=2)
            format_selector = gr.Radio(
                choices=["PNG", "JPG", "PT"],
                value="PNG",
                label="📁 Download Format",
                scale=1
            )
            save_btn = gr.Button("💾 Save & Download", variant="stop", size="lg", scale=1)

        # Status and Download
        with gr.Row():
            status_text = gr.Textbox(label="📊 Status", interactive=False, lines=3, scale=2)
            download_file = gr.File(label="📥 Download", visible=False, scale=1)

        # State to store points and masks
        points_state = gr.State([])
        masks_data = gr.State({"masks": [], "scores": [], "image": None})  # Store all mask data
        current_mask_index = gr.State(0)  # Current mask being viewed

        # Event handlers
        def on_image_click(image, current_points, negative_mode, evt: gr.SelectData):
            """Handle clicks on the image for point annotations only."""
            if evt.index is not None and image is not None:
                x, y = evt.index
                try:
                    pil_image = fix_image_array(image)
                    is_negative = negative_mode
                    new_point = {"x": int(x), "y": int(y), "positive": not is_negative}
                    updated_points = current_points + [new_point]

                    positive_points = [[p["x"], p["y"]] for p in updated_points if p.get("positive", True)]
                    negative_points = [[p["x"], p["y"]] for p in updated_points if not p.get("positive", True)]

                    updated_visualization = create_original_with_input_visualization(
                        pil_image, positive_points, None, "Points", negative_points
                    )

                    point_type = "positive" if not is_negative else "negative"
                    pos_count = len(positive_points)
                    neg_count = len(negative_points)
                    return updated_points, updated_points, updated_visualization, (
                        f"Added {point_type} point at ({x}, {y}). Total: {pos_count} positive, {neg_count} negative points."
                    )
                except Exception as e:
                    print(f"Error in visualization: {e}")
                    return current_points, current_points, None, f"Error updating visualization: {str(e)}"
            return current_points, current_points, None, "Click on the image to add points."

        def on_image_upload(image):
            """Handle image upload and show it for annotation."""
            if image is not None:
                try:
                    pil_image = fix_image_array(image)
                    img_array = np.array(pil_image)
                    # Populate both the annotation image (left) and the points preview (right)
                    return img_array, img_array, [], [], "Image uploaded. Click on the left image to add points (enable negative mode for exclusion)."
                except Exception as e:
                    return None, None, [], [], f"Error loading image: {str(e)}"
            return None, None, [], [], "No image uploaded."

        def clear_all_points(image):
            """Clear points and keep the image visible for annotation."""
            try:
                if image is not None:
                    pil_image = fix_image_array(image)
                    img_array = np.array(pil_image)
                    return [], [], img_array, img_array, None, "All points cleared. You can continue annotating."
            except Exception:
                pass
            return [], [], None, None, None, "All points cleared."

        def clear_text_prompt():
            """Clear the text prompt."""
            return "", "Text prompt cleared. You can now use manual points."

        def generate_segmentation(image, points, text_prompt, num_masks_to_generate):
            """Generate multiple segmentation masks - auto-detects input type."""
            # Determine image name
            if isinstance(image, str):
                image_name = os.path.basename(image)
            else:
                # Prefer an explicit friendly default if metadata lacks a good name
                image_name = None
                if hasattr(image, 'orig_name'):
                    image_name = image.orig_name
                elif isinstance(image, dict) and 'orig_name' in image:
                    image_name = image['orig_name']
                elif hasattr(image, 'name'):
                    image_name = image.name
                if not image_name or 'tmp' in str(image_name).lower() or 'uploaded_image' in str(image_name).lower():
                    image_name = "michael_phelps_bottom_left.jpg"

            # Auto-detect input type and run segmentation
            has_text = text_prompt and text_prompt.strip()
            has_points = points and len(points) > 0
            
            if has_text and has_points:
                # Combine text detection with manual point refinement
                status_info = "🎯 Combining text detection with manual point refinement"
                
                # First, detect with text to get initial bounding box
                detected_boxes, detection_status = detect_objects_with_text(image, text_prompt, 0.25)
                if detected_boxes:
                    # Use the detected bounding box AND manual points together
                    bbox = detected_boxes[0]  # Use first detection as guidance
                    
                    # Process with both bounding box and points
                    # The points will be used to refine the segmentation within the detected area
                    _, mask_img, status, masks, scores = process_sam_segmentation(
                        image, points, bbox, "Points", image_name, int(num_masks_to_generate), 0.0, 0.95, 0, None, 0.25
                    )
                    status = f"{status_info}\n✓ Text: {detection_status}\n✓ Using {len(points)} manual points for refinement\n{status}"
                    masks_data_dict = {"masks": masks, "scores": scores, "image": image}
                    return mask_img, status, masks_data_dict, 0, f"Mask 1 of {len(masks)} (Score: {scores[0]:.3f})"
                else:
                    # Fall back to points only if text detection fails
                    _, mask_img, status, masks, scores = process_sam_segmentation(
                        image, points, None, "Points", image_name, int(num_masks_to_generate), 0.0, 0.95, 0, None, 0.25
                    )
                    status = f"🔄 Text detection failed, using {len(points)} manual points only\n{status}"
                    masks_data_dict = {"masks": masks, "scores": scores, "image": image}
                    return mask_img, status, masks_data_dict, 0, f"Mask 1 of {len(masks)} (Score: {scores[0]:.3f})"
            elif has_text:
                # Use text prompt
                _, mask_img, status, masks, scores = process_sam_segmentation(
                    image, None, None, "Text Prompt", image_name, int(num_masks_to_generate), 0.0, 0.95, 0, text_prompt, 0.25
                )
                masks_data_dict = {"masks": masks, "scores": scores, "image": image}
                return mask_img, status, masks_data_dict, 0, f"Mask 1 of {len(masks)} (Score: {scores[0]:.3f})"
            elif has_points:
                # Use points
                _, mask_img, status, masks, scores = process_sam_segmentation(
                    image, points, None, "Points", image_name, int(num_masks_to_generate), 0.0, 0.95, 0, None, 0.25
                )
                masks_data_dict = {"masks": masks, "scores": scores, "image": image}
                return mask_img, status, masks_data_dict, 0, f"Mask 1 of {len(masks)} (Score: {scores[0]:.3f})"
            else:
                return None, "❌ Please either enter a text prompt or click points on the image.", {"masks": [], "scores": [], "image": None}, 0, "No masks generated"

        def navigate_mask(direction, current_index, masks_data):
            """Navigate through generated masks"""
            masks = masks_data.get("masks", [])
            scores = masks_data.get("scores", [])
            image = masks_data.get("image", None)
            
            if not masks or len(masks) == 0:
                return None, current_index, "No masks available"
            
            # Calculate new index
            if direction == "next":
                new_index = (current_index + 1) % len(masks)
            else:  # previous
                new_index = (current_index - 1) % len(masks)
            
            # Get the mask at new index
            mask = masks[new_index]
            score = scores[new_index]
            
            # Update global state for saving
            global CURRENT_MASK
            CURRENT_MASK = mask
            
            # Create visualization
            if image is not None:
                pil_image = fix_image_array(image)
                mask_visualization = create_mask_visualization(pil_image, mask, score, 0.0)
            else:
                mask_visualization = None
            
            mask_info_text = f"Mask {new_index + 1} of {len(masks)} (Score: {score:.3f})"
            
            return mask_visualization, new_index, mask_info_text

        def save_and_download_mask(custom_folder_name, download_format):
            """Save mask locally and prepare download for user."""
            global CURRENT_MASK, CURRENT_IMAGE_NAME, CURRENT_POINTS
            if CURRENT_MASK is None:
                return "❌ No mask to save. Generate a mask first.", None
            if CURRENT_POINTS is None:
                return "❌ No points available. Generate a mask first.", None
            
            try:
                # Save locally (keep existing hierarchy)
                local_save_status = save_binary_mask(
                    CURRENT_MASK, CURRENT_IMAGE_NAME, CURRENT_POINTS, 
                    0.0, 0, False, custom_folder_name=(custom_folder_name or None)
                )
                
                # Create download file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(CURRENT_IMAGE_NAME or "mask"))[0]
                
                if download_format == "PNG":
                    # Create PNG for download
                    binary_mask = (CURRENT_MASK > 0).astype(np.uint8) * 255
                    mask_image = Image.fromarray(binary_mask, mode='L')
                    download_path = f"/tmp/mask_{base_name}_{timestamp}.png"
                    mask_image.save(download_path, format="PNG")
                    
                elif download_format == "JPG":
                    # Create JPG for download
                    binary_mask = (CURRENT_MASK > 0).astype(np.uint8) * 255
                    mask_image = Image.fromarray(binary_mask, mode='L')
                    download_path = f"/tmp/mask_{base_name}_{timestamp}.jpg"
                    mask_image.save(download_path, format="JPEG", quality=95)
                    
                elif download_format == "PT":
                    # Create PyTorch tensor for download
                    download_path = f"/tmp/mask_{base_name}_{timestamp}.pt"
                    torch.save(torch.from_numpy((CURRENT_MASK > 0).astype(np.float32)), download_path)
                
                # Make download visible and return file
                download_status = f"✅ {local_save_status}\n📥 Download ready: {download_format} format"
                return download_status, gr.File.update(value=download_path, visible=True)
                
            except Exception as e:
                return f"❌ Save/download failed: {str(e)}", None

        # Wire events
        # Let the annotatable image also handle image uploads (drag & drop / click upload)
        original_with_input.upload(
            on_image_upload,
            inputs=[original_with_input],
            outputs=[original_with_input, points_overlay, points_state, points_display, status_text]
        )

        original_with_input.select(
            on_image_click,
            inputs=[original_with_input, points_state, negative_point_mode],
            outputs=[points_state, points_display, points_overlay, status_text]
        )

        # Generate button and Enter key support
        generate_btn.click(
            generate_segmentation,
            inputs=[original_with_input, points_state, text_prompt_input, num_masks],
            outputs=[mask_result, status_text, masks_data, current_mask_index, mask_info]
        )
        
        # Enter key support for text prompt
        text_prompt_input.submit(
            generate_segmentation,
            inputs=[original_with_input, points_state, text_prompt_input, num_masks],
            outputs=[mask_result, status_text, masks_data, current_mask_index, mask_info]
        )
        
        # Mask navigation
        prev_mask_btn.click(
            lambda idx, data: navigate_mask("prev", idx, data),
            inputs=[current_mask_index, masks_data],
            outputs=[mask_result, current_mask_index, mask_info]
        )
        
        next_mask_btn.click(
            lambda idx, data: navigate_mask("next", idx, data),
            inputs=[current_mask_index, masks_data],
            outputs=[mask_result, current_mask_index, mask_info]
        )

        clear_points_btn.click(
            clear_all_points,
            inputs=[original_with_input],
            outputs=[points_state, points_display, points_overlay, original_with_input, mask_result, status_text]
        )
        
        clear_text_btn.click(
            clear_text_prompt,
            outputs=[text_prompt_input, status_text]
        )

        save_btn.click(
            save_and_download_mask,
            inputs=[mask_name_input, format_selector],
            outputs=[status_text, download_file]
        )
    
    return interface

def main():
    """Main function"""
    print("🚀 Starting Fixed SAM 2.1 Interface...")
    
    interface = create_interface()
    
    print("🌐 Launching web interface...")
    print("📍 Click on objects in images to segment them!")
    
    interface.launch(
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=True,  # Enable public sharing
        inbrowser=False,  # Don't auto-open browser in server environment
        show_error=True,
        server_name="0.0.0.0",  # Allow external connections
        auth=None  # No authentication for public access
    )

if __name__ == "__main__":
    main()