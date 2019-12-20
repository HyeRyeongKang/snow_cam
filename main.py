import cv2, dlib, sys
import numpy as np

scaler=0.3

#얼굴 디텍터 모듈 초기화
detector=dlib.get_frontal_face_detector()
#얼굴 특징점 모듈 초기화
#머신러닝으로 학습된 모델 파일
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#load video
cap=cv2.VideoCapture('face.mp4')
#file이미지를 BGRA타입으로 읽기
overlay=cv2.imread('ryan_transparent.png', cv2.IMREAD_UNCHANGED)

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img


while True:
    ret, img=cap.read()
    if not ret:
        break

    img=cv2.resize(img, (int(img.shape[1]*scaler), int(img.shape[0]*scaler)))
    origin=img.copy()

    #detect faces
    #img에서 모든 얼굴 찾기
    faces=detector(img)
    face=faces[0]   #0번 인덱스 얼굴만 face에 저장

    #얼굴 특징점 추출
    dlib_shape=predictor(img, face)
    #dlib객체를 numpy 객체로 변환
    shape_2d=np.array([[p.x, p.y] for p in dlib_shape.parts()])

    #compute center of face
    top_left=np.min(shape_2d, axis=0)    #최소값 찾기
    bottom_right=np.max(shape_2d, axis=0)   #최대값 찾기

    #mean() : 평균 구하기
    center_x, center_y=np.mean(shape_2d, axis=0).astype(np.int)

    face_size=int(max(bottom_right-top_left)*1.8)
    result=overlay_transparent(origin, overlay, center_x, center_y, overlay_size=(face_size, face_size))

    #visualize
    #얼굴 영역 사각형으로
    img=cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255), 
    thickness=2, lineType=cv2.LINE_AA)

    #원그리기
    #특징점 68개
    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.imshow('result', result)
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)