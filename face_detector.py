#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 초기화
# =========================================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Face Detection 모델 초기화
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,             # 모델 선택 (0: 빠름, 1: 정확)
    min_detection_confidence=0.5   # 탐지 신뢰도
)

# =========================================
# 📸 카메라 연결
# =========================================
# cap = cv2.VideoCapture(0)            # 기본 카메라 사용시
# 동영상 파일 이름을 'face.mp4'로 설정했습니다.
cap = cv2.VideoCapture("face.mp4")   # 동영상 파일 사용 시 

print("📷 카메라 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. 카메라 연결을 확인하거나 동영상을 종료합니다.")
        break

    # 좌우 반전 (셀카 뷰 - 선택 사항)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환 (MediaPipe는 RGB 입력을 선호)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_detection.process(image_rgb)

    # 🧑‍ 얼굴 영역 표시
    if result.detections:
        for detection in result.detections:
            # 감지된 얼굴 주변에 경계 상자와 6개의 키포인트(눈, 귀, 코, 입) 그리기
            mp_drawing.draw_detection(
                image,
                detection,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),      # 경계 상자 스타일 (초록색)
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)       # 키포인트 스타일 (빨간색)
            )

    # 화면 표시
    cv2.imshow('🧑‍ MediaPipe Face Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()