import cv2
import os
import csv
import pandas as pd

def save_results(output_image, measurements):
    # 결과 이미지 저장
    if not os.path.exists("results"):
        os.makedirs("results")
    cv2.imwrite("results/output.jpg", output_image)

    # 측정 결과 저장 (CSV)
    with open("results/measurements.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Object", "Width (cm)", "Height (cm)"])
        writer.writerows(measurements)

def analyze_statistics(measurements):
    # 통계 분석 함수
    df = pd.DataFrame(measurements, columns=['Object', 'Width (cm)', 'Height (cm)'])
    
    stats = df[['Width (cm)', 'Height (cm)']].describe()
    stats.to_csv('results/statistics.csv')
    
    return stats
