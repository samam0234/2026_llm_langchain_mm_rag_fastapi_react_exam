import pytesseract       # OCR 라이브러리
from PIL import Image    # 이미지 파일 열기

# ── Windows 전용 설정 ─────────────────────────────────────────
# Linux/macOS는 이 줄 없어도 됩니다.
# tesseract.exe가 설치된 경로를 정확히 적어야 합니다.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = Image.open('images/capture3.jpg').convert('RGB')

result = pytesseract.image_to_string(img, lang='kor')

print(result.strip())