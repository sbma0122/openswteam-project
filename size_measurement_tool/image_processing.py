import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    # 1. 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Matplotlib 호환 RGB 포맷

    # 2. 사용자 기준 물체 선택
    print("기준 물체를 선택하세요: 마우스로 드래그한 후 Enter 또는 Spacebar를 누르세요.")
    roi = cv2.selectROI("Select Reference Object", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Reference Object")  # ROI 선택 후 창 닫기

    # 3. 기준 물체 영역에서 스케일 계산
    x, y, w, h = map(int, roi)
    reference_object = image[y:y + h, x:x + w]  # 기준 물체의 ROI 이미지
    ref_width = w  # 기준 물체 너비 (픽셀)
    actual_width = float(input("기준 물체의 실제 너비(cm)를 입력하세요: "))  # 실제 너비 입력

    scale = actual_width / ref_width  # 스케일 계산

    #update commit here!!
    # 4. 다중 스케일 객체 탐지
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    measurements = []
    
    # 이미지 피라미드 생성
    scales = [1.0, 0.75, 0.5]
    for scale_factor in scales:
        resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
        
        _, thresh = cv2.threshold(resized, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 원본 이미지 크기로 좌표 변환
            x, y = int(x / scale_factor), int(y / scale_factor)
            w, h = int(w / scale_factor), int(h / scale_factor)
            
            width_cm = w * scale
            height_cm = h * scale
            
            # 너무 작은 객체 무시
            if width_cm > 1 and height_cm > 1:
                measurements.append((width_cm, height_cm))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #finish my commit!!

    # 5. 객체 크기 측정 및 시각화
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
