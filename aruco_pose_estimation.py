import cv2
import cv2.aruco as aruco
import numpy as np
import os
import glob
import yaml

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_size= 200

def generate_marker(aruco_dict,marker_size):
	# Generate Aruco marker into png files
	folder_name = "markers"
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	for i in range(50):
		marker_id=i
		marker = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
		marker_filename = f"marker_{marker_id}.png"
		cv2.imshow("marker"+str(i),marker)
		cv2.imwrite(os.path.join(folder_name,marker_filename), marker)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def make_marker_board(folder_name = "markers"):
	# Combine all markers into a board
	marker_list = os.listdir(folder_name)
	marker_list = sorted(marker_list)
	num_marker = len(marker_list)
	print("number of markers",num_marker)
	img_temp = cv2.imread(os.path.join(folder_name,marker_list[0]))
	width,height = img_temp.shape[0:2]

	# Add margins between markers
	board_image = np.ones((int(num_marker/10*(40+height)),10*(40+width),3))*255
	for ind,marker in enumerate(marker_list):
		if marker!="board.png":
			ind_row = int(ind/10)
			ind_col = int(ind%10)
			board_image[ind_row*(40+height)+20:(ind_row+1)*(40+height)-20,ind_col*(40+width)+20:(ind_col+1)*(40+width)-20,:]=cv2.imread(os.path.join(folder_name , marker))
	cv2.imshow("board",board_image)
	cv2.waitKey(0)
	cv2.imwrite(os.path.join(folder_name,"board.png"),board_image)
	print("board saved")

def capturePictures():
	# Press "s" key to capture photos
	folder_name="captured_photo"
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	num_pictures = 14
	num_id = 0
	filename = "cap_pic"
	cap = cv2.VideoCapture(0)
	while cap.isOpened() and num_pictures>0 :
		ret,frame=cap.read()
		if not ret:
			break
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame',gray)
		key = cv2.waitKey(1)
		if key==ord("s"):
			cv2.imwrite(os.path.join(folder_name,filename+f"_{num_id}.jpg"),gray)
			num_pictures -=1
			print("Picture "+str(num_id)+" saved."+str(num_pictures)+" more to take.")
			num_id+=1
		elif key ==ord("q"):
			break
	cap.release()
	cv2.destroyAllWindows()


def calibrateCamera():
	# Calibrate camera with photo of chessboards

	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.
	
	images = glob.glob('captured_photo/*.jpg')
	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)
			corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
			imgpoints.append(corners2)
			cv2.drawChessboardCorners(img, (9,6), corners2, ret)
			cv2.imshow('img', img)
			cv2.waitKey(5000)
	cv2.destroyAllWindows()
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	#print("mtx",mtx)
	#print("dist",dist)

	# Display undistorted images
	#for fname in images:
	#	img = cv2.imread(fname)
	#	h, w = img.shape[:2]
	#	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
	#	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
	#	x, y, w, h = roi
	#	dst = dst[y:y+h, x:x+w]
	#	cv2.imshow('img', img)
	#	cv2.waitKey(5000)
	# crop the image
	#cv2.destroyAllWindows()
	
	# save camera matrix into yaml file
	calibration_data = {'camera_matrix': mtx.tolist(),'distortion_coefficients': dist.tolist()}
	with open('laptop_webcam_calibration_data.yaml', 'w') as file:
		yaml.dump(calibration_data, file)
	print("Calibration data saved to 'laptop_webcam_calibration_data.yaml'")
	

def load_calibration_data(yaml_file="laptop_webcam_calibration_data.yaml"):
	with open(yaml_file, 'r') as file:
	    calibration_data = yaml.safe_load(file)

	# Extract the camera matrix and distortion coefficients
	mtx = np.array(calibration_data['camera_matrix'])
	dist = np.array(calibration_data['distortion_coefficients'])

	# Print the loaded calibration data
	print("Loaded Camera Matrix:")
	print(mtx)
	print("Loaded Distortion Coefficients:")
	print(dist)
	return mtx,dist


def detect_markers(aruco_dict):
	parameters =  aruco.DetectorParameters()
	mtx,dist = load_calibration_data()
	markerLength = 0.05
	# Set the coordinates of the marker corners
	objPoints = np.array([[-markerLength/2, markerLength/2, 0],
                      [markerLength/2, markerLength/2, 0],
                      [markerLength/2, -markerLength/2, 0],
                      [-markerLength/2, -markerLength/2, 0]], dtype=np.float32)
	cap = cv2.VideoCapture(0)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print("width,height",width,height)

	while cap.isOpened():
		ret,frame = cap.read()
		if not ret:
			break
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
		if ids is not None:
			rvecs,tvecs = [],[]
			marked = aruco.drawDetectedMarkers(frame.copy(), corners, ids, (0,255,0))
			for ind,marker_id in enumerate(ids):
				print(f"id:{marker_id},corners{corners[ind]}")
				_,rvec, tvec= cv2.solvePnP(objPoints, corners[ind], mtx, dist)
				rvecs.append(rvec)
				tvecs.append(tvec)
			for ind in range(len(ids)):
				axe_image = cv2.drawFrameAxes(marked, mtx, dist, rvecs[ind], tvecs[ind], 0.1)
			cv2.imshow('frame',axe_image)
		else:
			cv2.imshow('frame',frame)
		key = cv2.waitKey(1)
		if key==ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__=="__main__":
	#generate_marker(aruco_dict,marker_size)
	#make_marker_board()
	#calibrateCamera()
	#capturePictures()
	detect_markers(aruco_dict)