# 데이터 라벨링이 잘못되면 훈련 결과가 비참해진다.
# 훈련전에 라벨을 자동 체크하여 최악의 경우를 미리 예방한다.

import os, glob
import re

def check_labels(label_dir, num_classes):
    """
    YOLO 라벨 파일이 규칙에 맞는지 검사한다 (초보자 디버깅용).

    [검사 항목]
        1) 한 줄은 정확히 5개 값이어야 한다 (클래스 + 좌표4개)
        2) 클래스 번호는 0 이상 num_classes 미만의 정수여야 한다
        3) 좌표 4개는 모두 0~1 사이여야 한다 (정규화 규칙)
    [반환] 문제점 문자열 리스트. 비어있으면 정상.
    """
    problems = []
    tmp_path = os.path.join(label_dir, "*.txt")
    tmp_path = re.sub(r"\\", "/", tmp_path)  # 윈도우 경로 구분자 문제 해결
    print(f"라벨 검사: {tmp_path}")
    for path in glob.glob(os.path.join(label_dir, "*.txt")):
        print(f"검사 파일: {path}")
        for i, line in enumerate(open(path), 1):    # 한 라인씩 읽어서
            line = line.strip()
            if not line:
                continue                            # 빈 줄은 건너뜀
            parts = line.split()
            if len(parts) != 5:                     # ① 값 개수 검사
                problems.append(f"{path}:{i} 값 개수 {len(parts)} (5개여야 함)")
                continue
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
            if not (0 <= cls < num_classes):        # ② 클래스 범위 검사
                problems.append(f"{path}:{i} class {cls} 범위초과")
            if any(v < 0 or v > 1 for v in vals):   # ③ 좌표 범위 검사
                problems.append(f"{path}:{i} 좌표가 0~1 범위 밖")
    return problems

# 사용 예
bad = check_labels("D:/Qwerting(Langchain)/ai-cctv-platform/part05_yolo", num_classes=1)
print("정상 라벨" if not bad else f"문제 {len(bad)}건: {bad}")