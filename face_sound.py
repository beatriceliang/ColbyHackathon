import cv2
import sound
faceCascade = cv2.CascadeClassifier("./faces/haarcascade_frontalface_alt.xml")
video_capture = cv2.VideoCapture(0)

#returns normalized location of face in image (x,y)
def get_face_loc(frame, face):
    print "Face:",face[0],face[1]
    print "Frame:", frame.shape[0], frame.shape[1]
    return  float(face[0]+face[2])/frame.shape[1], float(face[1]+face[3])/frame.shape[0]
def face_sound(frame, face):
    return
def main():
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
       
        for face in faces:
            print get_face_loc(frame, face)   
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()