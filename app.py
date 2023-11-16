from fastapi import FastAPI, HTTPException, UploadFile
import mediapipe as mp
import numpy as np
import cv2 
from keras.models import load_model
import uvicorn
import base64

def initialize():
    return load_model('model'), mp.solutions.pose.Pose()

model, pose = initialize()


# Constants
LABELS = {0: "Back Lunges", 1: "Box Jumps", 2: "Glute Bridges", 3: "Overhead Squat"}

MUSCLES = {
    "Overhead Squat": ["Quadriceps", "Hamstrings", "Glutes", "Lower Back", "Core"],
    "Box Jumps": ["Quadriceps", "Hamstrings", "Calves", "Glutes"],
    "Glute Bridges": ["Glutes", "Hamstrings", "Lower Back"],
    "Back Lunges": ["Quadriceps", "Hamstrings", "Glutes", "Calves"]
}

EXCLUDED_INDICES = [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,31,32]

app = FastAPI()
def process_landmarks(frame): 
    global pose
    return pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def classify_exercise(results, model):
    landmarks = [landmark for i, landmark in enumerate(results.pose_landmarks.landmark) if i not in EXCLUDED_INDICES]
    inputs = np.array([val for landmark in landmarks for val in (landmark.x, landmark.y, landmark.z, landmark.visibility)]).reshape(1, -1)
    prediction = model.predict(inputs)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = LABELS[predicted_class_index]
    predicted_probability = prediction[0, predicted_class_index]
    return predicted_class_label, predicted_probability*100

def save_pose_skeleton(results, frame):
    if results:
        h, w, c = frame.shape 
        dummy_img = np.ones([h, w, c])
        dummy_img.fill(255)
            
        mp.solutions.drawing_utils.draw_landmarks(dummy_img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                        mp.solutions.drawing_utils.DrawingSpec(color=(245,0,255), thickness=8, circle_radius=2),
                                        mp.solutions.drawing_utils.DrawingSpec(color=(245,117,0), thickness=10, circle_radius=2))
    return dummy_img

# Endpoint for exercise classification
@app.post("/predict")
async def classify_exercise_endpoint(file: UploadFile):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = process_landmarks(frame)

        class_name, accuracy = classify_exercise(results, model)
        muscles = MUSCLES[class_name]

        skeleton_image = save_pose_skeleton(results, frame)
        _, skeleton_image_buffer = cv2.imencode(".png", skeleton_image)
        skeleton_image_base64 = base64.b64encode(skeleton_image_buffer).decode("utf-8")
        return {"exercise_type": class_name, "Accuracy": f"{accuracy}%", "muscles_involved": muscles, "skeleton_image": skeleton_image_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the uploaded file")

# Root endpoint
@app.get('/')
async def root():
    return {"about": "Explore the Exercise Classifier API to identify and learn about various exercises effortlessly", "version":"0.0.1"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5738)
