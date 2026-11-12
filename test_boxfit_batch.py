
from pathlib import Path
from test_boxfit_integrated_v1 import analyze_image

cases = {
    "six_boxes": "test_images/test_B_top.png",
    "one_box": "test_images/test_A.png",
    "empty_box": "test_images/test_D_top.png",
}

for name, path in cases.items():
    print(f"\n=== {name} ===")

    try:
        result = analyze_image(
            Path(path),
            Path(f"test_outputs/batch_{name}.png")
        )

        print("제품 점유율:", result["occupancy_percent"], "%")
        print("빈 면적률:", result["void_percent"], "%")
        print("바닥 좌표:", result["floor"])
        print("결과 이미지:", result["output"])

    except Exception as e:
        print("분석 실패:", str(e))
