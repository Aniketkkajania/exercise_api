import cv2
import requests
import numpy as np
import base64

# Define the API URL
api_url = "http://127.0.0.1:5734/classify-exercise"

# Open the video capture
cap = cv2.VideoCapture("C:/Users/abhis/Downloads/exercise_model/male-dumbbell-overhead-squat-front.mp4")

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to JPEG format
    _, buffer = cv2.imencode(".jpg", frame)
    jpg_image = buffer.tobytes()

    # Send the frame to the API for classification
    files = {"file": ("frame.jpg", jpg_image, "image/jpeg")}
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        data = response.json()
        exercise = data["exercise"]
        muscles = data["muscles"]
        skeleton_image_base64 = data["skeleton_image"]

        # Decode the base64 skeleton image
        skeleton_image = base64.b64decode(skeleton_image_base64)
        skeleton_image_np = np.frombuffer(skeleton_image, np.uint8)
        skeleton_frame = cv2.imdecode(skeleton_image_np, cv2.IMREAD_COLOR)

        # Save the skeleton image to the output video
        cv2.imshow("Frame", skeleton_frame)
        cv2.waitKey(10)  # Adjust the wait time as needed

        print(f"Frame {frame_number}: Exercise: {exercise}, Muscles: {', '.join(muscles)}")
    else:
        print(f"Frame {frame_number}: Failed to classify exercise")

    frame_number += 1

# Release the video writer and video capture
cap.release()
cv2.destroyAllWindows()

