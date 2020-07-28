import numpy as np
from cv2 import cv2
import time

from find_corners_of_largest_polygon import find_corners_of_largest_polygon
from sudoku_grid import  distance_between, crop_and_warp, infer_grid, display_rects
from digit_extractor import get_digits
from norvig import solve, display

starttime = time.time()

# webcam feed
cap = cv2.VideoCapture(2)
ret, original_frame = cap.read()
height, width, _ = original_frame.shape
out = cv2.VideoWriter('/home/sahil/Char74kModel/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))

# cap = cv2.VideoCapture('/home/sahil/Char74kModel/temp.mp4')


# iterate through the frames
while cap.isOpened():

    
     # Read the frame and preprocess
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # # cap.set(cv2.CAP_PROP_BUFFERSIZE, 3);
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ret, original_frame = cap.read()
    out.write(original_frame)

   
    # height = 480
    # width=640

    
    #we will use this later

    gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    # Blurring the unwanted stuff
    blur = cv2.GaussianBlur(gray, (9,9), 0)


    # Adaptive threshold for to obtain a Binary Image from the frame
    # followed by Hough lines probablistic transform to highlight the straight line proposals
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed = cv2.bitwise_not(adaptive, adaptive)
    # processed_not_dilated = processed.cop

    # Dilate the image to increase the size of the grid lines.
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],dtype=np.uint8)
    processed = cv2.dilate(processed, kernel)


    # Find the proposals sudoku contour- Contours are all posssible points. We will create the grid using contours
    contours, hier = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    external_contours = cv2.drawContours(processed.copy(), contours, -1, (255, 0, 0), 2)
    
    current_time = (time.time()- starttime)
    if current_time > 1.0:
        starttime = time.time()
        #We are assuming the user is not nifty and he is not trying to solve the pandora box
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
        try:
            sudoku_square = contours[0] 
        except Exception as e:
            print("Camera stopped working!")
            break
            cv2.destroyAllWindows()
            cap.release()
            


        corners = find_corners_of_largest_polygon(sudoku_square)
        #We found the corners of the sudoku now and pointed them.

        #Now the image might be sheared. We have to convert it to a perfect square
        cropped = crop_and_warp(original_frame, corners)
        final_image = cropped.copy()

        sudoku_coordinates = np.array([[0, 0], [cropped.shape[0]-1,0], [cropped.shape[0]-1, cropped.shape[1]-1], [0,cropped.shape[1]-1]])   

        squares = infer_grid(cropped)

        side = cropped.shape[:1]
        side = side[0] / 9
        #fetch digit boxes
        digits = get_digits(cropped, squares, 28)
    

        if len(digits) == 81:
            digits = np.array(digits).reshape(9, 9 , 28, 28)
            digits = np.transpose(digits, (1,0,2,3)).reshape(81,28,28,1)

            grid, solved = solve(digits)
            if grid ==".................................................................................":
                continue
            
            if solved != False:
                
                digits = np.array(digits).reshape(9, 9 , 28, 28)
                digits = np.transpose(digits, (1,0,2,3)).reshape(81,28,28)
                display(solved)
                solved = list((np.array(list(solved.values())).reshape(9,9).T).reshape(81))
                
            
                # Print the text on our image
            
                for number in range(81):
                    cv2.imwrite("/home/sahil/Char74kModel/snippets/"+str(number)+'.png',digits[number].reshape(28,28))
                    
                    if digits[number].any() != 0:
                        continue
                
                    bottom = squares[number][0][0] + 0.25 * side
                    left = squares[number][1][1] - 0.25 * side
                    final_image=cv2.putText(final_image, solved[number], ( int(bottom), int(left) ),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), thickness=1)
                    
                    
                cv2.imwrite('/home/sahil/Char74kModel/Final_sol.png', final_image)
                h, mask = cv2.findHomography(sudoku_coordinates, np.array(corners))
                final_image = cv2.warpPerspective(final_image, h, (width, height))
                final_image =cv2.addWeighted(final_image, 0.5, original_frame, 0.5, 1)
                cv2.imshow('Sudoku solver', final_image)
                for i in range(5000):
                    out.write(final_image)
                break
            #Stop the webcam now and wait 5 seconds to disply the output. Also save the image
                
    cv2.imshow('Sudoku solver', original_frame)

    # Exit on ESC
    if (cv2.waitKey(1) == 27):
        break
cv2.waitKey(50000)
cv2.destroyAllWindows()
out.release()
cap.release()



