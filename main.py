import cv2
import numpy as np
import stackImages as stack

class AirDraw():
    def __init__(self):
        #super(self).__init()
        self.x,self.y,self.isSelected = 200, 200, False
        self.cap = cv2.VideoCapture(0)

## To get the optical Point selected by mouse pointer
    def mouse_pointer_position(self, event, x1, y1, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x = x1
            self.y = y1
            self.isSelected = True

## Drawing circles/lines from previous point to new point from the specified optical mark
    def opticalFlow(self, img, gray_img):
        stamp = 0

        old_pts = np.array([[self.x, self.y]], dtype=np.float32).reshape(-1,1,2)

        mask = np.zeros_like(img)

        while True:
            _, new_img = self.cap.read()
            new_img = cv2.flip(new_img, 1)
            new_gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)     
            new_pts,status,err = cv2.calcOpticalFlowPyrLK(gray_img, 
                                new_gray_img, 
                                old_pts, 
                                None, maxLevel=1,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                15, 0.08))

            for i, j in zip(new_pts, old_pts):
                x,y = i.ravel()
                a,b = j.ravel()
                if cv2.waitKey(2) & 0xff == ord('r'):
                    stamp = 1
                    
                elif cv2.waitKey(2) & 0xff == ord('s'):
                    stamp = 0
                
                elif cv2.waitKey(2) == ord('n'):
                    mask = np.zeros_like(new_img)
                    
                if stamp == 0:
                    mask = cv2.line(mask, (a,b), (x,y), (0,0,255), 6)

                cv2.circle(new_img, (x,y), 6, (0,255,0), -1)
            
            new_img = cv2.addWeighted(mask, 0.3, new_img, 0.7, 0)
            cv2.putText(mask, "Long_Press 'r' -> relese, 's' -> start, 'n' -> clear, 'esc' -> exit ", (10,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255),1)
            # cv2.imshow("ouput", new_img)
            # cv2.imshow("result", mask)
            imgResult = stack.stackImages(1.0, ([new_img, mask]))
            cv2.imshow("Air Paint", imgResult)

            
            gray_img = new_gray_img.copy()
            old_pts = new_pts.reshape(-1,1,2)
            
            if cv2.waitKey(1) & 0xff == 27:
                break

        cv2.destroyAllWindows()
        self.cap.release()

## Getting the optical mark/point from the mouse_pointer_position function and getting started
    def main(self):

        cv2.namedWindow("Select_Point")
        cv2.setMouseCallback("Select_Point", self.mouse_pointer_position)

        while True:
            _, img = self.cap.read()
            img = cv2.flip(img, 1)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            cv2.putText(img,"Select A point", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,20,20), 2)
            cv2.imshow("Select_Point", img)
            
            if cv2.waitKey(30) == 27:
                break
            if self.isSelected == True:
                self.opticalFlow(img, gray_img)
                break
        
        cv2.destroyAllWindows()
        
    
    


## making a instence of the AirDraw class and calling the main function to start the program
airDraw = AirDraw()
if __name__ == "__main__":
    airDraw.main()