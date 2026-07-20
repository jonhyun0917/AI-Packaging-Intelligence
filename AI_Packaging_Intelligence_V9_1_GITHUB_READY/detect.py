# detect.py
# AI Packaging Intelligence System V4.1
# Product Segmentation Stable Version


import cv2
import numpy as np



def detect_products(image, box=None):


    img = np.array(image)


    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )


    result_image = img.copy()



    h,w = img.shape[:2]



    # ==========================
    # Box Crop
    # ==========================


    if box:


        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]


    else:

        x1=int(w*0.05)
        y1=int(h*0.05)

        x2=int(w*0.95)
        y2=int(h*0.95)



    crop = img[y1:y2,x1:x2]



    crop_h,crop_w=crop.shape[:2]


    crop_area=crop_h*crop_w



    # ==========================
    # Segmentation
    # ==========================


    gray=cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )


    blur=cv2.GaussianBlur(
        gray,
        (11,11),
        0
    )


    edges=cv2.Canny(
        blur,
        40,
        130
    )



    kernel=np.ones(
        (11,11),
        np.uint8
    )


    edges=cv2.dilate(
        edges,
        kernel,
        iterations=1
    )


    edges=cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )



    contours,_=cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )



    candidates=[]



    for cnt in contours:


        area=cv2.contourArea(cnt)



        # 작은 노이즈 제거
        if area < crop_area*0.008:

            continue



        # 박스 전체 제거
        if area > crop_area*0.4:

            continue



        px,py,pw,ph=cv2.boundingRect(cnt)



        ratio=pw/ph if ph else 0



        if ratio < 0.15 or ratio > 6:

            continue



        rect_area=pw*ph


        fill_ratio=area/rect_area



        # 너무 빈 영역 제거
        if fill_ratio < 0.15:

            continue



        center_x=px+pw/2
        center_y=py+ph/2



        distance=(

            abs(center_x-crop_w/2)

            +

            abs(center_y-crop_h/2)

        )



        # 면적 중심 선택

        score=(

            area*0.8

            -

            distance*10

        )

        # 후보 면적 비율 계산

        area_ratio = area / crop_area


        # 비슷한 크기의 중복 후보 제거 기준

        if area_ratio < 0.02:
            continue



        candidates.append({

            "score":score,

            "area":area,

            "bbox":(

                px,

                py,

                pw,

                ph

            )

        })



    detections=[]

    product_area=0

    total_products=0



    # ==========================
    # Best Candidate
    # ==========================


    if candidates:


        candidates.sort(

            key=lambda x:x["score"],

            reverse=True

        )


        best=candidates[0]



        px,py,pw,ph=best["bbox"]



        rx1=px+x1
        ry1=py+y1

        rx2=px+pw+x1
        ry2=py+ph+y1



        product_area=int(

            best["area"]

        )


        total_products=1



        detections.append({

            "name":"Detected Product",

            "confidence":0.9,

            "area":product_area,

            "bbox":[

                rx1,

                ry1,

                rx2,

                ry2

            ]

        })



        cv2.rectangle(

            result_image,

            (rx1,ry1),

            (rx2,ry2),

            (0,255,0),

            3

        )



        cv2.putText(

            result_image,

            "product 0.90",

            (rx1,ry1-10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0,255,0),

            2

        )



    print("===================")

    print("Product Segmentation V4.1")

    print(detections)

    print(

        "Product Area:",

        product_area

    )

    print(

        "Candidates:",

        len(candidates)

    )

    print("===================")



    return (

        detections,

        result_image,

        product_area,

        total_products

    )
