import cv2
import os

def train_face_recognizer(data_dir):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []

    for name in os.listdir(data_dir):
        label = int(name)
        image_dir = os.path.join(data_dir, name)
        for filename in os.listdir(image_dir):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(label)

    face_recognizer.train(faces, np.array(labels))
    return face_recognizer

def recognize_faces(image_path, face_recognizer):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = image[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi)
        print(f"Label: {label}, Confidence: {confidence}")

        # Save the recognized faces with bounding boxes
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, f"Label: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Save the result image
    result_image_path = "result.jpg"
    cv2.imwrite(result_image_path, image)

if __name__ == "__main__":
    data_dir = "dataset"  # Replace with the path to your dataset
    image_path = "predictions.jpg"  # Replace with the path to the test image
    face_recognizer = train_face_recognizer(data_dir)
    recognize_faces(image_path, face_recognizer)
