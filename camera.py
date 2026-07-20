# camera.py

import cv2



def camera_start():


    cap=cv2.VideoCapture(0)


    while True:


        ret,frame=cap.read()


        if not ret:

            break



        cv2.imshow(

            "AI Packaging Camera",

            frame

        )



        key=cv2.waitKey(1)



        if key==27:

            break



    cap.release()


    cv2.destroyAllWindows()
    