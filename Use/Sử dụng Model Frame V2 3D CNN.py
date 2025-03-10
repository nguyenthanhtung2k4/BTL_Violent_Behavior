import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model("fight.keras")

# Define image size and frames per file (match model input)
img_size = 224  # Update img_size to match your model's input size
_images_per_file = 20  # Update frames per file to match the model's input


def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    # Reshape to add an extra channel dimension for grayscale conversion
    img = img.astype(np.float32)  # Convert the image to float32 type
    img = np.expand_dims(img, axis=0)

    return img


def predict_violence_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = []
    frame_sequence = []  # Store frames for the sequence

    while success and len(frames) < _images_per_file:
        frames.append(image)
        processed_frame = preprocess_frame(image)  # Extract transfer values
        frame_sequence.append(processed_frame)
        success, image = vidcap.read()

    if not frames:
        print(f"Error: No frames could be read from {video_path}.")
        return

    # Pad with the last frame if fewer than _images_per_file frames
    while len(frame_sequence) < _images_per_file:
        frame_sequence.append(frame_sequence[-1])

    # Prepare input for the 3D CNN model
    input_sequence = np.concatenate(frame_sequence, axis=0)  # Shape: (_images_per_file, img_size, img_size, 3)
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(input_sequence)
    violence_probability = prediction[0][0]  # Get violence probability

    print(f"Overall Violence Probability for the video: {violence_probability * 100:.2f}%")

    # Process each frame and print individual violence probabilities
    for i, frame in enumerate(frames):
        # Use preprocessed frames from frame_sequence
        frame_data = frame_sequence[i]
        # Create dummy sequence with repeated frame if needed
        dummy_sequence = np.repeat(frame_data, _images_per_file, axis=0)
        dummy_sequence = np.expand_dims(dummy_sequence, axis=0)

        frame_prediction = model.predict(dummy_sequence)
        frame_violence_prob = frame_prediction[0][0]

        print(f"Frame {i + 1}: Violence Probability = {frame_violence_prob * 100:.2f}%")

        # Display the frame with its violence probability
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Frame {i + 1}: Violence Probability: {frame_violence_prob * 100:.2f}%")
        plt.show()


# Example usage with a video
video_path = "camera.mp4"  # Replace with the actual video path
predict_violence_video(video_path)