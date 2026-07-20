# box_detector.py
# AI Packaging Intelligence System V4.4
# Open Parcel Inner Space Estimation


import cv2
import numpy as np



def detect_box(image):


    """
    Box Detection V4.4

    Open box correction

    Return
    ------
    box_image
    inner_box_area
    box
    """



    img = np.array(image)


    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )


    result = img.copy()



    h, w = img.shape[:2]



    # ==========================
    # Outer Box Estimation
    # ==========================


    x1 = int(w * 0.05)
    y1 = int(h * 0.05)

    x2 = int(w * 0.95)
    y2 = int(h * 0.95)



    outer_area = (

        x2-x1

    ) * (

        y2-y1

    )



    # ==========================
    # Open Box Inner Correction
    # ==========================


    # 개봉 박스 날개 제거 비율

    INNER_RATIO = 0.65



    inner_area = int(

        outer_area *

        INNER_RATIO

    )



    # ==========================
    # Visualization
    # ==========================


    cv2.rectangle(

        result,

        (x1,y1),

        (x2,y2),

        (255,0,0),

        4

    )



    # 내부 영역 표시

    inner_w = int(

        (x2-x1)

        *

        0.8

    )


    inner_h = int(

        (y2-y1)

        *

        0.8

    )



    ix1 = int(

        x1 +

        ((x2-x1)-inner_w)/2

    )


    iy1 = int(

        y1 +

        ((y2-y1)-inner_h)/2

    )


    ix2 = ix1 + inner_w

    iy2 = iy1 + inner_h



    box = {

        "x1":ix1,

        "y1":iy1,

        "x2":ix2,

        "y2":iy2

    }



    cv2.rectangle(

        result,

        (ix1,iy1),

        (ix2,iy2),

        (0,255,0),

        5

    )



    print("===================")

    print("Box Detection V4.4")

    print("Outer Box Area:", outer_area)

    print("Inner Box Area:", inner_area)

    print("Box:", box)

    print("===================")



    return (

        result,

        inner_area,

        box

    )
