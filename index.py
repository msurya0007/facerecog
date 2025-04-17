import cv2
import numpy as np
import os
import xlrd
import xlwt
from xlutils.copy import copy as xl_copy
from datetime import date

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup
current_folder = os.getcwd()
known_faces = {'surya': cv2.imread(os.path.join(current_folder, 'surya.png'), 0),
                'varma': cv2.imread(os.path.join(current_folder, 'varma.png'), 0)}

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Ensure attendance file exists
attendance_file = 'attendance_excel.xls'
if not os.path.exists(attendance_file):
    wb = xlwt.Workbook()
    wb.add_sheet('Sheet1')
    wb.save(attendance_file)

# Setup attendance sheet
rb = xlrd.open_workbook(attendance_file, formatting_info=True)
wb = xl_copy(rb)
sheet_name = "Lecture_Attendance"  # Default sheet name to avoid input error
sheet = wb.add_sheet(sheet_name)
sheet.write(0, 0, 'Name/Date')
sheet.write(0, 1, str(date.today()))
row = 1

attendance_taken = set()

while True:
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        name = "Unknown"
        
        for known_name, known_face in known_faces.items():
            res = cv2.matchTemplate(face_roi, known_face, cv2.TM_CCOEFF_NORMED)
            if np.max(res) > 0.6:  # Threshold for matching
                name = known_name
                break
        
        if name != "Unknown" and name not in attendance_taken:
            sheet.write(row, 0, name)
            sheet.write(row, 1, "Present")
            row += 1
            attendance_taken.add(name)
            wb.save(attendance_file)
            print(f"Attendance taken for {name}")
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
