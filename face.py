import cv2
import face_recognition
import os
import numpy as np
import pickle
import threading
from pathlib import Path
import win32file
import win32api

DB_PATH = Path(r"C:\Users\임지민\Documents\high school\science festival\celebrity_images\archive\img_align_celeba\img_align_celeba")
ENCODINGS_PATH = Path("face_encodings_optimized.pkl")

def load_database():
    if ENCODINGS_PATH.exists():
        print("기존 데이터베이스 로딩 중...")
        with ENCODINGS_PATH.open('rb') as f:
            return pickle.load(f)
    else:
        print("데이터베이스 파일을 찾을 수 없습니다.")
        return None

def find_most_similar(face_encoding, database, tolerance=0.6):
    best_match = None
    best_similarity = 0
    for filename, encoding in database.items():
        distance = face_recognition.face_distance([encoding], face_encoding)[0]
        if distance <= tolerance and (1 - distance) > best_similarity:
            best_match = filename
            best_similarity = 1 - distance
    return best_match, best_similarity

def process_frame(frame, face_locations):
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    return frame

def face_recognition_thread(frame, database, result):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    result['locations'] = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]
    result['encodings'] = face_encodings

def read_image_win32(file_path):
    try:
        handle = win32file.CreateFile(str(file_path), win32file.GENERIC_READ, 
                                      win32file.FILE_SHARE_READ, None, 
                                      win32file.OPEN_EXISTING, 0, None)
        data = win32file.ReadFile(handle, 1024*1024)[1]  # Read up to 1MB
        win32file.CloseHandle(handle)
        return np.frombuffer(data, dtype=np.uint8)
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def show_similar_face(similar_image_path, similarity):
    if similar_image_path.exists():
        img_data = read_image_win32(similar_image_path)
        if img_data is not None:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (400, 400))
                cv2.putText(img, f"Similarity: {similarity:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Similar Face', img)
                cv2.waitKey(0)
                cv2.destroyWindow('Similar Face')
            else:
                print(f"이미지 디코딩 실패: {similar_image_path}")
        else:
            print(f"이미지 데이터를 읽을 수 없습니다: {similar_image_path}")
    else:
        print(f"이미지 파일을 찾을 수 없습니다: {similar_image_path}")

def main():
    print("프로그램 시작")
    database = load_database()
    if database is None:
        print("데이터베이스 로딩 실패. 프로그램을 종료합니다.")
        return

    print(f"데이터베이스에 {len(database)}개의 얼굴이 로드되었습니다.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return

    face_recognition_result = {'locations': [], 'encodings': []}
    face_recognition_thread_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 프레임을 읽을 수 없습니다.")
            break

        if not face_recognition_thread_active:
            face_recognition_thread_active = True
            thread = threading.Thread(target=face_recognition_thread, args=(frame, database, face_recognition_result))
            thread.start()

        processed_frame = process_frame(frame, face_recognition_result['locations'])
        cv2.imshow('Video', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13:  # Enter key
            if face_recognition_result['encodings']:
                similar_filename, similarity = find_most_similar(face_recognition_result['encodings'][0], database)
                if similar_filename:
                    similar_image_path = DB_PATH / similar_filename
                    print(f"가장 닮은 얼굴 찾음. 유사도: {similarity:.2f}")
                    show_similar_face(similar_image_path, similarity)
                else:
                    print("닮은 얼굴을 찾을 수 없습니다.")
            else:
                print("인식된 얼굴이 없습니다.")

        if not thread.is_alive():
            face_recognition_thread_active = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()