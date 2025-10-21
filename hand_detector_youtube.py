#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import yt_dlp
import sys

# =========================================
# # YouTube 스트림 URL 추출 함수
# =========================================
def get_youtube_stream_url(url):
    # YouTube 링크로부터 OpenCV에서 재생 가능한 mp4 스트림 URL을 추출
    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4]/best', # mp4 형식 우선 선택
        'noplaylist': True,
        'skip_download': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # mp4 형식 중 가장 좋은 화질의 URL을 찾아서 반환
        return info['url']

# =========================================
# 🎬 YouTube 영상 주소
# =========================================
youtube_url = "https://www.youtube.com/watch?v=wU7hEYvGZdU" 

print(f"🔗 YouTube 영상을 위한 스트림 URL을 가져오는 중...")
try:
    video_stream_url = get_youtube_stream_url(youtube_url)
    print(f"✅ 스트림 URL 추출 완료")
except Exception as e:
    print(f"❌ 오류: YouTube 스트림 URL을 가져올 수 없습니다. {e}")
    sys.exit()

# =========================================
# 🧩 Mediapipe 초기화
# =========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,       # 동영상 입력
    max_num_hands=2,              # 최대 손 개수
    min_detection_confidence=0.5, # 탐지 신뢰도
    min_tracking_confidence=0.5   # 추적 신뢰도
)

# =========================================
# 📹 YouTube 영상 열기
# =========================================
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("❌ YouTube 영상을 열 수 없습니다.")
    sys.exit()

print("✅ YouTube 스트림 재생 시작 - ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 스트림이 끊어졌거나 프레임을 읽지 못했습니다.")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 손 검출 수행
    result = hands.process(image_rgb)

    # 🖐️ 손 랜드마크 표시
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    # 결과 화면 표시
    cv2.imshow('🖐️ MediaPipe Hand Detector (YouTube)', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()