from image_processing import process_image
from utils import save_results

def main():
    # 1. 사용자로부터 이미지 파일 경로 입력
    image_path = input("측정할 이미지 파일 경로를 입력하세요: ")

    # 2. 이미지 처리 함수 호출
    output_image, measurements = process_image(image_path)

    # 3. 결과 저장
    save_results(output_image, measurements)

    print("측정 완료! 결과 이미지는 results 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
