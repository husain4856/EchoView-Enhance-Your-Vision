# EchoView-Enhance-Your-Vision
EchoView is a vision assistance app for visually impaired users. It provides scene understanding, object detection, OCR-based text extraction, and task-specific guidance. Using AI and text-to-speech, it helps users interact with their surroundings and navigate tasks through real-time image analysis.

# EchoView - Vision Assistance App

EchoView is a vision assistance app designed for visually impaired users. It leverages AI to analyze images in real-time and provide the following features:
- **Scene Understanding**: Describes the image context, objects, activities, and safety considerations.
- **Object Detection**: Detects and labels objects in an image with bounding boxes.
- **OCR & Text-to-Speech**: Extracts text from images and reads it aloud.
- **Task-Specific Guidance**: Suggests tasks and guides users on how to perform them based on the scene.

## Features
- **Real-time scene description**: Uses AI to describe images clearly and empathetically.
- **Object detection**: Identifies and labels common objects using a pre-trained Faster R-CNN model.
- **Text extraction**: Utilizes OCR to extract text and convert it to speech for accessibility.
- **Task guidance**: Generates actionable tasks or instructions based on the scene description.

## Requirements
- Python 3.x
- Streamlit
- PyTorch
- PIL
- pytesseract
- langchain-google-genai
- pyttsx3
- inflect
