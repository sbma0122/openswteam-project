import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    # 1. 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Matplotlib은 RGB 포맷 사용

    # 2. 이미지 전처리 (그레이스케일, 임계값 처리 등)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # 3. 객체 탐지 (윤곽선 검출)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 기준 물체 선택 및 스케일 계산
    reference_object = contours[0]  # 첫 번째 윤곽선을 기준 물체로 가정
    ref_width = cv2.boundingRect(reference_object)[2]  # 기준 물체 너비 (픽셀)
    actual_width = 2.0  # 실제 기준 물체 너비 (cm) - 사용자 정의

    scale = actual_width / ref_width

    # 5. 다른 객체 크기 측정
    measurements = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        width_cm = w * scale
        height_cm = h * scale
        measurements.append((width_cm, height_cm))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 6. Matplotlib으로 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title("Object Size Measurement")
    plt.axis("off")
    plt.show()

    return image, measurements
