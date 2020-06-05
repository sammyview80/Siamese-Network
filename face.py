import cv2
import os
from main import Preprocessing


def live():
    P = Preprocessing()
    P.setup()
    P.read_file()


    os.environ['DISPLAY'] = ':0'

    # initializing th classifiers
    cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)


    while True:
        # Capture frame by frame
        ret, frames = video_capture.read()

        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Getting the image 
            img = frames[y+5:y+h+5, x+5:x+w+5]

            prediction = P.compare(img)
            print(f'p: {prediction}')

            if prediction is not None:
                cv2.putText(frames, str(prediction), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video', frames)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # print(P.read_file())
    live()
