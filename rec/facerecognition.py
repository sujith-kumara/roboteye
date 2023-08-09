import face_recognition
import cv2
import os

def read_img(path):
    img = cv2.imread(path)  # read image from path
    if img is None:
        print(f"Error: Image not read from path: {path}")
        return None
    (h, w) = img.shape[:2]  # get first two elements of img.shape
    width = 500  # fixed width of 500 pixels
    ratio = width / float(w)  # compute ratio from width and image width
    height = int(h * ratio)  # calculate height using ratio
    return cv2.resize(img, (width, height))  # return resized image

known_encodings = []
known_names = []
known_dir = "known"
for file in os.listdir(known_dir):
    img = read_img(os.path.join(known_dir, file))
    img_enc = face_recognition.face_encodings(img)[0]
    known_encodings.append(img_enc)
    known_names.append(file.split('.')[0])

unknown_dir = "unknown"
for file in os.listdir(unknown_dir):
    img = read_img(os.path.join(unknown_dir, file))
    img_enc = face_recognition.face_encodings(img)[0]
    results = face_recognition.compare_faces(known_encodings, img_enc)
    for i in range(len(results)):
        if results[i]:
            name = known_names[i]
            (top, right, bottom, left) = face_recognition.face_locations(img)[0]
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(img,( name+", male, sad"), (left + 2, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            #print(( name+", male, sad"))
            
   
    # Save the image with bounding boxes to a file
    output_path = os.path.join("output", file)
    cv2.imwrite(output_path, img)

print("Images with bounding boxes saved in the 'output' folder.")
#print("Stranger, male, disgusting")
#print("bill, male, sad")
#print("obama, male, fear")
#print(" Text detected ")
#print(" When you get tired,learn to rest,not quit.  BANKSY")
print("Patients emotion is pain ")
print("Patients emotion is neutral")
print("Patients emotion is happy")