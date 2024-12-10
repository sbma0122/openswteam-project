import cv2
import numpy as np
from ultralytics import YOLO

def process_image(image_path, confidence_threshold=0.7, nms_threshold=0.4):
    # 이미지 로드
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise Exception("이미지를 불러올 수 없습니다.")

    # 이미지 전처리
    processed_image = preprocess_image(original_image.copy())

    # 기준 물체 선택
    reference_object = select_reference_object(processed_image)

    # 실제 너비 입력 받기
    real_width = float(input("선택한 기준 물체의 실제 너비(cm)를 입력하세요: "))

    # 픽셀당 실제 거리 계산
    pixels_per_cm = reference_object[2] / real_width

    # YOLO 모델 로드
    model = YOLO('yolov8n.pt')

    # 객체 탐지
    results = model(processed_image)

    # 측정 결과 저장
    measurements = []
    output_image = original_image.copy()

    # 결과 시각화 및 측정값 계산
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls)
            conf = float(box.conf)
            
            if conf < confidence_threshold:
                continue

            # 기준 물체와 동일한 객체는 제외
            if (x1, y1, x2-x1, y2-y1) == reference_object:
                continue

            # 실제 크기 계산 (cm)
            width_cm = (x2 - x1) / pixels_per_cm
            height_cm = (y2 - y1) / pixels_per_cm

            # 클래스 이름 가져오기
            class_name = model.names[class_id]

            # 측정값 저장
            measurements.append([class_name, width_cm, height_cm])

            # 바운딩 박스 그리기
            cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(output_image, f"{class_name}: {width_cm:.2f}x{height_cm:.2f}cm", 
                        (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return output_image, measurements

def preprocess_image(image):
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 이미지 정규화
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    
    # 대비 향상
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def select_reference_object(image):
    # 기준 물체 선택을 위한 ROI
    roi = cv2.selectROI("Select a reference object", image, False)
    cv2.destroyWindow("Select a reference object")
    return roi

def detect_objects(image):
    # 객체 탐지를 위한 기본 설정
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        # 면적이 너무 작은 컨투어 무시
        if cv2.contourArea(contour) < 100:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        confidence = calculate_confidence(contour)  # 컨투어 기반 신뢰도 계산
        
        detections.append({
            'bbox': (x, y, w, h),
            'confidence': confidence
        })
    
    return detections

def calculate_confidence(contour):
    # 컨투어의 면적과 둘레를 기반으로 신뢰도 계산
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return min(1.0, circularity)

def filter_detections(detections, confidence_threshold):
    return [det for det in detections if det['confidence'] > confidence_threshold]

def apply_nms(detections, nms_threshold):
    if not detections:
        return []
        
    boxes = [det['bbox'] for det in detections]
    scores = [det['confidence'] for det in detections]
    
    return cv2.dnn.NMSBoxes(
        boxes,
        scores,
        0.0,  # score_threshold
        nms_threshold
    ).flatten()
