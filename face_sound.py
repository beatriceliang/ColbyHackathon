import cv2
import sound
import pyaudio
import numpy as np
import threading
import colorsys

class Trail:
    def __init__(self,face, color):
        self.face = face
        self.color = color
        self.life = 10
        self.size = 0.5

#returns normalized location of face in image (x,y)
def get_face_loc(frame, face):
    return  float(face[0]+face[2])/frame.shape[1], float(face[1]+face[3])/frame.shape[0]

#maps the y location of the face to tones 14 to 42
def map2sound(frame, face):
    x,y = get_face_loc(frame,face)
    return int(29*(1-y)+14)

def get_sound(frame, faces):
    root = sound.Note('A',3)
    scale = sound.Scale(root, [2, 2, 1, 2, 2, 2, 1])
    return np.sum([sound.pluck2(scale.get(map2sound(frame, face))) for face in faces], axis=0)


def playSound(frame, faces, pyaud):
    stream = pyaud.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)
    stream.write(get_sound(frame, faces).astype(np.float32).tostring())
    stream.close()


def main():

    faceCascade = cv2.CascadeClassifier("./faces/haarcascade_frontalface_alt.xml")
    video_capture = cv2.VideoCapture(0)
    p = pyaudio.PyAudio()

    trails = []

    count = 0
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
        for i in range(len(faces)):
            x = faces[i][0]
            y = faces[i][1]
            w = faces[i][2]
            h = faces[i][3]
            color = colorsys.hsv_to_rgb(get_face_loc(frame,(x,y,w,h))[1],1,1)
            color = (color[2]*255,color[1]*255, color[0]*255)
            cv2.circle(frame, (x+w/2,y+h/2), w/2,color,-1 )            
            if len(trails)-1 < i:
                trails.append([Trail(faces[i],color)])
            else:
                trails[i].append(Trail(faces[i],color))
        
        for trail in trails:
            if len(trail) > 0:
                if trail[0].life == 0:
                    trail.pop(0)
            width = trail[len(trail)-1].face[2]
            for loc in trail:
                x = loc.face[0]
                y = loc.face[1]
                w = loc.face[2]
                h = loc.face[3]
                loc.life -=1
                loc.size-=0.03
                cv2.circle(frame, (x+w/2,y+h/2), int(width*loc.size),loc.color,-1 )

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Get the sound and add it to the list
        if count % 2000 == 0 and len(faces) > 0:
            thread = threading.Thread(target=playSound, args=(frame, faces, p))
            thread.daemon = True
            thread.start()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    p.terminate()

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()