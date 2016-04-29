import cv2
import sound
import numpy as np

#returns normalized location of face in image (x,y)
def get_face_loc(frame, face):
    return  float(face[0]+face[2])/frame.shape[1], float(face[1]+face[3])/frame.shape[0]

#maps the y location of the face to tones 14 to 42
def map2sound(frame, face):
    x,y = get_face_loc(frame,face)
    return int(29*(1-y)+14)

def get_sound(frame, faces):
    root = sound.Note('A',3)
    scale = sound.Scale(root,[2, 2, 1, 2, 2, 2, 1])

    chunks = []
    tones =np.empty((88200,1))
    for face in faces:
        tones+sound.pluck2(scale.get(map2sound(frame,face)))
        #chunks.append(sound.pluck2(sound.scale.get(map2sound(frame,face))))


def main():
    faceCascade = cv2.CascadeClassifier("./faces/haarcascade_frontalface_alt.xml")
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draws a dot at the center of the faces
        for (x, y, w, h) in faces:
            cv2.circle(frame, (x+w/2,y+h/2), 2,(0, 255, 0),2 )
        get_sound(frame, faces)
        for face in faces:
            print map2sound(frame,face)
            # print get_face_loc(frame, face)   
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()