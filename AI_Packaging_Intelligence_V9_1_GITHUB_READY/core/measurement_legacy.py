"""
Pixel → Real Size Conversion
"""


def calculate_scale(
        pixel_length,
        real_cm
):

    """
    pixel/cm 계산
    """

    return pixel_length / real_cm





def pixel_to_cm(
        pixel,
        scale
):

    return round(
        pixel / scale,
        2
    )





def bbox_to_size(
        bbox,
        scale
):

    x1,y1,x2,y2=bbox


    width=x2-x1

    height=y2-y1



    return {

        "width":
            pixel_to_cm(
                width,
                scale
            ),

        "height":
            pixel_to_cm(
                height,
                scale
            )

    }
