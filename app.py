import torch
from ultralytics import YOLO
import gradio as gr
from collections import Counter

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

def detect_objects(image):
    """
    Detects objects in an image using the YOLO model and returns a count of each object.
    Args:
        image: A PIL Image or a NumPy array representing the image.
    Returns:
        A dictionary where keys are object names and values are their counts.
    """
    results = model(image)

    detected_objects = []
    for r in results:
        if r.boxes is not None:  # Check if any objects were detected
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                detected_objects.append(class_name)

    object_counts = Counter(detected_objects)
    return object_counts

def format_output(object_counts):
    """
    Formats the object counts dictionary into a neat string.
    Args:
        object_counts: A dictionary of object names and their counts.
    Returns:
        A formatted string representing the object counts.
    """
    output_string = ""
    for obj, count in object_counts.items():
        output_string += f"{obj.capitalize()} - {count}\n"
    return output_string

def predict(image):
    """
    Combines object detection and output formatting for the Gradio interface.
    Args:
        image: The input image from the Gradio interface.
    Returns:
        A formatted string of detected objects and their counts.
    """
    object_counts = detect_objects(image)
    formatted_output = format_output(object_counts)
    return formatted_output

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload an Image"),
    outputs=gr.Textbox(label="Detected Objects and Counts"),
    title="Object Detection with YOLOv8",
    description="Upload an image and I'll tell you what objects I see!",
)

# Launch the Gradio app
iface.launch()