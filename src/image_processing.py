import cv2


class ImageProcessing:

    def __init__(self):
        pass

    def image_processing(self, frame, kernel, fgbg):
        a = 0
        pred = 0
        bounding_rect = []  # An empty list where will furthur input the contours

        # Applying the changes of the background to the foreground mask
        fgmask = fgbg.apply(frame)

        fgmask = cv2.erode(fgmask, kernel, iterations=5)
        fgmask = cv2.dilate(fgmask, kernel,iterations=5)  # Erosion and Dilation is done to detect even the blur objects better

        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   # The mode RETR_TREE together with the CHAIN APPROX_SIMPLE returns only the endpoints required to draw contour

        for i in range(0, len(contours)):
            bounding_rect.append(cv2.boundingRect(contours[i]))  # cv2.bounding rectangle gives the coordinates of bounding rectangle and then we will input these coordinates to the list we made
        for i in range(0, len(contours)):
            if bounding_rect[i][2] >= 40 or bounding_rect[i][3] >= 40:  # setting the threshold for the width and height if the contour
                a = a + (bounding_rect[i][2]) * bounding_rect[i][3]  # updating the area of contour
            if a >= int(frame.shape[0]) * int(frame.shape[1]) / 3:  # It is basically the comparison of the change in area of the background, so if there is a certain change in area it will detect the tampering
                pred = 3  # Moved

        return pred