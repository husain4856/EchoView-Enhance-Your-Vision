import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import pyttsx3
import pytesseract
import io
import base64
import inflect

# Initialize Chat model
chat_model = ChatGoogleGenerativeAI(api_key='AIzaSyAtR-MZMPI91mGIdi34SSPriUiI8wFUbjg', model="gemini-1.5-flash")

# Load Faster R-CNN for object detection
@st.cache_resource
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

object_detection_model = load_object_detection_model()

# COCO class labels (91 categories)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Draw bounding boxes on image
def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    labels = predictions['labels']
    boxes = predictions['boxes']
    scores = predictions['scores']

    for label, box, score in zip(labels, boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label.item()]  # Map label ID to class name
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="black")
    return image

# Object Detection Function
def detect_objects(image, threshold=0.5, iou_threshold=0.5):
    inflect_engine = inflect.engine()
    try:
        # Transform the image into a tensor
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image)
        
        # Get predictions from the model
        predictions = object_detection_model([img_tensor])[0]
        
        # Apply Non-Maximum Suppression to filter out overlapping boxes
        keep = torch.ops.torchvision.nms(predictions['boxes'], predictions['scores'], iou_threshold)
        
        # Filter predictions based on score threshold
        filtered_predictions = {
            'boxes': predictions['boxes'][keep],
            'labels': predictions['labels'][keep],
            'scores': predictions['scores'][keep]
        }

        # List to store detected objects
        detected_objects = []

        for label in filtered_predictions['labels']:
            class_name = COCO_CLASSES[label.item()]  
            detected_objects.append(class_name)
        
        # Count occurrences of each object
        object_counts = {}
        for obj in detected_objects:
            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        # Prepare final detected object list with count and pluralization
        final_detected_objects = []
        for obj, count in object_counts.items():
            # Use inflect to get the plural form of the object name
            pluralized_obj = inflect_engine.plural_noun(obj) if count > 1 else obj
            final_detected_objects.append(f"{pluralized_obj}")
        
        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image.copy(), filtered_predictions, threshold)
        
        return final_detected_objects, image_with_boxes
    except Exception as e:
        raise RuntimeError(f"Failed to detect objects. Error: {e}")


# Text-to-Speech Function
def text_to_speech(text, x=False):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1)
    audio_file = "text-to-speech-local.mp3"
    try:
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        st.audio(audio_file,format="audio/mp3")
    except Exception as e:
        st.error(f"Audio generation failed: {e}")

# OCR and Text-to-Speech
def text_extraction_and_audio(uploaded_image):
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        extracted_text = pytesseract.image_to_string(uploaded_image)
        if extracted_text.strip():
            st.write("Extracted Text:")
            st.write(extracted_text)
            text_to_speech(extracted_text)
        else:
            st.write("No text found in the image.")
            text_to_speech("No text found in the image.")
    except Exception as e:
        st.error(f"Text extraction failed: {e}")


# Task-Specific Guidance (modified to use chat_with_bot)
def task_specific_guidance(image):
    tasks = generate_tasks_from_description(image)
    
    if tasks:
        # Display the generated tasks for the user to select
        selected_task = st.radio("Select a task to perform:", tasks,index=1)
        
        if selected_task:
            st.write(f"You have selected: {selected_task}")
            
            # Send the selected task as a prompt to the chatbot
            hmessage = HumanMessage(content=[{"type": "text", "text": selected_task}])
            try:
                # Get the chatbot's response based on the selected task
                response = chat_model.invoke([hmessage])
                response_text = response.content
                st.write(response_text)
                text_to_speech(response_text)
            except Exception as e:
                st.error(f"Task guidance failed: {e}")
    else:
        st.error("No tasks generated from the description.")


# Function to generate 5 tasks/questions based on the description using ChatGoogleGenerativeAI
def generate_tasks_from_description(image):
    description = real_time_scene_understanding(image, chat = True)
    print(description)
    hmessage = HumanMessage(
        content=[{
            "type": "text",
            "text": f"""Based on the following image description, generate 3 tasks which you can help the user perform using the objects in the image or tasks which can be put into steps, do not generate questions which require you to take another image input 
            Image Description: {description}
            Only the questions should be generated nothing else.
            """
        }]
    )
    
    try:
        # Send the message to ChatGoogleGenerativeAI to generate tasks
        response = chat_model.invoke([hmessage])
        response_text = response.content
        tasks = response_text.split("\n")
        
        tasks = [task.strip() for task in tasks if task.strip()]
        
        if len(tasks) >= 5:
            return tasks[:5]  
        else:
            return tasks 
    except Exception as e:
        st.error(f"Failed to generate tasks. Error: {e}")
        return []



# Real-time scene understanding
def real_time_scene_understanding(image, chat = False):
    # Convert the uploaded image to a base64 string
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')  # Save image as PNG to ensure compatibility
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')  # Convert image bytes to base64

    # Prepare the message for Gemini API with the image as base64
    hmessage = HumanMessage(
        content=[{
            "type": "text",
            "text": """You are a real-time scene interpreter for visually impaired users. Your task is to analyze and describe images vividly, empathetically, and without technical jargon. Focus on delivering concise, actionable information that enhances understanding and safety.

            Description Guidelines:
            - Scene Overview: Summarize the setting (e.g., indoor/outdoor, type of location).
            - Key Objects & Layout: Describe objects with details like position, color, size, shape, and texture.
            - Activities & Interactions: Highlight actions or interactions (e.g., "A person jogging with a dog").
            - Mood & Atmosphere: Describe the tone (e.g., "Lively with bright sunlight").
            - Text & Symbols: Transcribe visible text or signs.
            - Sensory Details: Mention implied sounds, smells, or sensations.
            - Accessibility & Safety: Identify potential hazards or challenges (e.g., "A step down near the doorway").
            - Formatting Tips:
                - Use short, clear sentences.
                - Structure from general to specific details.
                - Maintain a neutral, empathetic tone."""},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
        ]
    )

    try:
        if chat == True :
            response = chat_model.invoke([hmessage])
            response_text = response.content
            return response_text
        response = chat_model.invoke([hmessage])
        response_text = response.content
        st.write(response_text)
        text_to_speech(response_text)
    except Exception as e:
        st.error(f"Scene understanding failed: {e}")

# Streamlit UI
st.title("EchoView: Enhance Your Vision")


if 'welcome_spoken' not in st.session_state:
    st.session_state['welcome_spoken'] = False

# Display Welcome message and speak it only once at the start
if not st.session_state['welcome_spoken']:
    welcome_text = """
    Welcome to the EchoView: Enhance Your Vision! 
    Simply upload an image and select one of the four tasks on the screen.
    If listening is your skill, our audio outputs will fit the bill! 
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('volume', 1)
    engine.say(welcome_text)
    engine.runAndWait()
    st.session_state['welcome_spoken'] = True

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🖼 Scene Understanding"):
        st.session_state['task'] = "Scene Understanding"

with col2:
    if st.button("📦 Object Detection"):
        st.session_state['task'] = "Object Detection"

with col3:
    if st.button("🔍 OCR and Text-to-Speech"):
        st.session_state['task'] = "OCR and Text-to-Speech"

with col4:
    if st.button("🧭 Task Specific Guidance"):  # Added symbol for Task Specific Guidance
        st.session_state['task'] = "Task-Specific Guidance"
        
if 'task' in st.session_state:
    st.success(f"You selected: {st.session_state['task']}")


# Instructions Section with Expander
st.markdown("### Instructions")
with st.expander("🔍 How to use this app"):
    st.markdown("""
        <div style="font-size: 25px; line-height: 1.6;">
            1. Upload an image using the button below.<br>
            2. Select one of the tasks: Scene Understanding, Object Detection,Text Extraction or Task Specific Guidance.<br>
            3. Wait for the results, which will be displayed on the screen.<br>
            4. You can also hear the results via audio.
        </div>
    """, unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload an image for analysis:", type=['jpg', 'jpeg', 'png'])

st.markdown("""
    <style>
    .stButton>button {
        background-color: #004d99; /* Dark Blue */
        color: #ffffff; /* White text */
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        width: 100%; 
    }
    .stButton>button:hover {
        background-color: #002b5c; /* Darker Blue */
    }
    </style>
""", unsafe_allow_html=True)


if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.session_state['task'] == "Scene Understanding":
        real_time_scene_understanding(image, chat = False)
    elif st.session_state['task'] == "OCR and Text-to-Speech":
        text_extraction_and_audio(image)
    elif st.session_state['task'] == "Task-Specific Guidance":
        task_specific_guidance(image)
    elif st.session_state['task'] == "Object Detection":
        detected_objects, image_with_boxes = detect_objects(image)
        st.image(image_with_boxes, caption="Object Detection Results", use_container_width=True)
        st.write("Detected Objects:", detected_objects)
        text_to_speech(detected_objects)
