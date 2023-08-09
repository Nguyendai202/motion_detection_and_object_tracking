import imutils# xử lí video 
import cv2
import numpy as np
import os 


# Create kalman filter object
def create_kalman():
    dt = 0.2
    kalman = cv2.KalmanFilter(4, 2, 0)
    kalman.transitionMatrix = np.array(
        [[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.processNoiseCov = 0.5 * np.array([[dt ** 4.0 / 4.0, 0., dt ** 3.0 / 2.0, 0.],
                                             [0., dt ** 4.0 / 4.0, 0., dt ** 3.0 / 2.0],
                                             [dt ** 3.0 / 2.0, 0., dt ** 2.0, 0.],
                                             [0., dt ** 3.0 / 2.0, 0., dt ** 2.0]], dtype=np.float32)
    kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
    kalman.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)

    return kalman


video_path = 'test.avi'

# Initialize video reader object
vs = cv2.VideoCapture(video_path)
save_dir = "video_outout"
height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)#default 480
width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)#default 640
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps = vs.get(cv2.CAP_PROP_FPS)# lấy khung hình mỗi s của video đầu vào 
fps = 10
save_path = 'dai_0308_30fps.avi'
save_path = os.path.join(save_dir,save_path)
writer = cv2.VideoWriter(save_path, fourcc,fps, (int(width), int(height)))# đầu ra giữ như đầu vào  
fgbg = cv2.createBackgroundSubtractorMOG2()

# Kalman Filter
# trong trường hợp phéo đo gmm ko xác định đc tham số thì nó vẫn có thể dự báo đc các giá trị tại thời điểm đó>tracking 
kalman = create_kalman()

# List detection by background subtraction
list_detection = []

# List estimation by kalman filter
list_predict = []

# Process
while True:
    # Read video
    _, frame = vs.read()

    # If end of video, break
    if frame is None:
        break

    # resize the frame, easy to display
    frame = imutils.resize(frame, width=500)

    # Apply GMM model to frame
    thresh = fgbg.apply(frame)

    # Remove noise
    thresh = cv2.medianBlur(thresh, 5)

    # Dilate
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Threshold image
    _, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)

    # Find contours(tìm các contours đc phân đoạn và lớn nhất )
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours with maximum area
    maxArea = 0
    maxContour = None
    for c in cnts:# nếu ko có contour nào tìm thấy thì thiết lập lại danh sách là rỗng 
        # nếu tìm thấy thì vẽ hình chữ nhật bao quanh 
        if (cv2.contourArea(c) > maxArea) and (cv2.contourArea(c) >= 500):
            maxArea = cv2.contourArea(c)
            maxContour = c

    if maxContour is None:
        list_detection = []

        # New
        list_predict = []

    else:
        #>>>>>>># Draw result(box bao quanh đối tượng , tìm đc tâm dx,dy của đối tượng>>measure)
        (x, y, w, h) = cv2.boundingRect(maxContour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dx = int(x + w / 2)
        dy = int(y + h / 2)
        list_detection.append((dx, dy))

        # Init statePre
        if len(list_detection) == 1:
            kalman = create_kalman()
            kalman.statePre = np.array([[dx], [dy], [0.], [0.]], dtype=np.float32)
 #>>>>>>>>> # Kalman predict(dựa vào trạng thái trước đó để dự báo)
        estimate = kalman.predict()
  #>>>>>>>> # Kalman correct: kết hợp measure với predict của kalman
  # trạng thái khi ước lượng dựa trên cái vị trí dự báo và phép đo đó và 
  # mong muốn nó trở nên đúng với vị trí thực tế nhất 
        estimated = kalman.correct(np.array([[dx], [dy]], dtype=np.float32))
        list_predict.append((estimated[0, 0], estimated[1, 0]))
        # cv2.circle(frame,(estimated[0,0],estimated[1,0]),3,(255,0,0),1)
        # cv2.circle(frame,(dx,dy),3,(0,0,255),1)
        if len(list_detection) > 1:# vẽ đường tracking
            for i in range(len(list_detection) - 1):
                x, y = list_detection[i]
                u, v = list_detection[i + 1]
                # cv2.line(frame, (x, y), (u, v), (0, 0, 255))

            # Draw tracker
            for i in range(len(list_detection) - 1):
                x, y = list_predict[i]
                u, v = list_predict[i + 1]
                # cv2.line(frame, (x, y), (u, v), (255, 0, 0))

    # Display result

    cv2.imshow("Original", frame)
    cv2.imshow("GMM", thresh)
    writer.write(frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_path = "img_crop.jpg"
        save_path = os.path.join(save_dir,save_path)
        cv2.imwrite(save_path,frame)
        print("An image is saved to ",save_path)

cv2.destroyAllWindows()
