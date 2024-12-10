from image_processing import process_image
from utils import save_results, analyze_statistics

def main():
    # 1. 사용자로부터 이미지 파일 경로 입력
    image_path = input("측정할 이미지 파일 경로를 입력하세요: ")

    # " 문자를 포함하는 경로 처리
    image_path = image_path.replace('"', '')

    # 2. 이미지 처리 함수 호출
    output_image, measurements = process_image(image_path)

    # 3. 결과 저장
    save_results(output_image, measurements)

    # 4. 통계 분석
    stats = analyze_statistics(measurements)
    print("통계 분석 결과:")
    print(stats)

    print("측정 완료! 결과 이미지와 통계 데이터는 results 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
