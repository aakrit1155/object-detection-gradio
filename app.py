import torch
from ultralytics import YOLO
import gradio as gr
from collections import Counter
from PIL import Image, ImageDraw
import os
import shutil
import numpy as np

MODEL_NAME = 'yolov8n.pt'  # Specify the model name

CUSTOM_PATH_TO_MODEL = os.path.join(os.path.dirname(__file__), 'models', MODEL_NAME)
DEFAULT_DOWNLOAD_PATH = os.path.join(os.path.dirname(__file__), 'models',MODEL_NAME) # Set to the default download path

# Load the YOLOv8 model from a custom path if it exists, otherwise load the default model
if CUSTOM_PATH_TO_MODEL != DEFAULT_DOWNLOAD_PATH:
    if os.path.exists(CUSTOM_PATH_TO_MODEL):
        print(f"Loading model from: {CUSTOM_PATH_TO_MODEL}")
        model = YOLO(CUSTOM_PATH_TO_MODEL)
    else:
        print(f"Model not found at: {CUSTOM_PATH_TO_MODEL}. Loading default (will download if needed).")
        model = YOLO(MODEL_NAME)
        
        import shutil
        shutil.move(DEFAULT_DOWNLOAD_PATH, CUSTOM_PATH_TO_MODEL)
else:
    print(f"Loading default model from: {DEFAULT_DOWNLOAD_PATH}")
    model = YOLO(MODEL_NAME)



def detect_objects(image):
    """
    Detects objects in an image using the YOLO model and returns the results object.
    Args:
        image: A PIL Image or a NumPy array representing the image.
    Returns:
        The YOLO results object containing detection information.
    """
    results = model(image)
    return results


def draw_bounding_boxes(image, results):
    """
    Draws bounding boxes and labels on the input image.
    Args:
        image: A PIL Image object.
        results: The YOLO results object.
    Returns:
        A PIL Image object with bounding boxes and labels drawn on it.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]} {conf:.2f}"

                x1, y1, x2, y2 = map(int, xyxy)
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                draw.text((x1, y1 - 10), label, fill="red")

    return image_with_boxes

def format_output(results):
    """
    Formats the object counts from the YOLO results into a neat string.
    Args:
        results: The YOLO results object.
    Returns:
        A formatted string representing the object counts.
    """
    detected_objects = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                detected_objects.append(class_name)

    object_counts = Counter(detected_objects)
    output_string = ""
    for obj, count in object_counts.items():
        output_string += f"{obj.capitalize()} - {count}\n"
    return output_string


def predict(image):
    """
    Combines object detection, bounding box drawing, and output formatting for the Gradio interface.
    Args:
        image: The input image from the Gradio interface.
    Returns:
        A tuple containing the image with bounding boxes and the formatted object counts.
    """
    results = detect_objects(image)
    image_with_boxes = draw_bounding_boxes(image, results)
    formatted_output = format_output(results)
    return image_with_boxes, formatted_output

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload an Image"),
    outputs=[
        gr.Image(label="Image with Detected Objects"),
        gr.Textbox(label="Detected Objects and Counts")
    ],
    title="Object Detection with YOLOv8",
    description="Upload an image and I'll show you the detected objects with bounding boxes and a list of objects with their counts!",
)

# Launch the Gradio app
iface.launch()