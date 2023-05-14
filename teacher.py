import os
import numpy as np
import rospy
import miro2 as miro
from std_msgs.msg import UInt16MultiArray, Int16MultiArray
import cv2
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message

class Teacher():
    TICK = 0.02
    CAM_FREQ = 1
    def __init__(self):

        # which miro
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # subscribers
        self.sub_mics = rospy.Subscriber(topic_base_name + "/sensors/mics",
            Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)

        self.sub_caml = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_caml,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_camr = rospy.Subscriber(
            topic_base_name + "/sensors/camr/compressed",
            CompressedImage,
            self.callback_camr,
            queue_size=1,
            tcp_nodelay=True,
        )
       
        # publishers
        self.beep_pub = rospy.Publisher(topic_base_name + "/control/tone",
            UInt16MultiArray, queue_size=1, tcp_nodelay=True)

        # Initialise CV Bridge
        self.image_converter = CvBridge()

        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of both cylinder's x, y, and r values for each camera
        self.cylinder = [None, None]
        self.cylinder_two = [None, None]
        # Set the default frame width (gets updated on receiving an image)
        self.frame_width = 640
        # Action selector to reduce duplicate printing to the terminal
        self.just_switched = True
        # Bookmark
        self.bookmark = 0
        self.instruction = 0
        self.missing = 0
        self.resultMessage = 0
        self.SIGNALTIME = 2
        message1 = UInt16MultiArray()
        message1.data = [1000,255,self.SIGNALTIME]
        message2 = UInt16MultiArray()
        message2.data = [1300,255,self.SIGNALTIME]
        message3 = UInt16MultiArray()
        message3.data = [1600,255,self.SIGNALTIME]
        message4 = UInt16MultiArray()
        message4.data = [2000,255,self.SIGNALTIME]
        self.messages = [message1,message2,message3,message4]
        self.greenSeen = True
        self.blueSeen = True
        self.frameMissingCount = 0
        self.DEBUG = True
        
        rospy.sleep(1.0)

    def callback_mics(self, tcp): 
        pass

    def callback_caml(self, ros_image):  # Left camera
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):  # Right camera
        self.callback_cam(ros_image, 1)

    def callback_cam(self, ros_image, index):
        """
        Callback function executed upon image arrival
        """
        # Silently(-ish) handle corrupted JPEG frames
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # Convert from OpenCV's default BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Store image as class attribute for further use
            image = image[170:190,1:640]
            self.input_camera[index] = image
            # Get image dimensions
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2
            self.y_centre = self.frame_height / 2
            # Crop image
            # Raise the flag: A new frame is available for processing
            self.new_frame[index] = True
        except CvBridgeError as e:
            # Ignore corrupted frames
            pass

    def detect_cylinder(self, frame, index, colour):
        if frame is None:  # Sanity check
            return

        # Debug window to show the frame
        if self.DEBUG:
            cv2.imshow("camera" + str(index), frame)
            cv2.waitKey(1)

        # Flag this frame as processed, so that it's not reused in case of lag
        self.new_frame[index] = False
        # Get image in HSV (hue, saturation, value) colour format
        im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # RGB values of target cylinder
        rgb_cylinder = [np.uint8([[[255, 0, 0]]]), np.uint8([[[0, 0, 255]]]), np.uint8([[[0, 255, 0]]])]  # e.g. Blue (Note: BGR)
        # Convert RGB values to HSV colour model
        hsv_cylinder = cv2.cvtColor(rgb_cylinder[colour], cv2.COLOR_RGB2HSV)        

        # Extract colour boundaries for masking image
        # Get the hue value from the numpy array containing target colour
        target_hue = hsv_cylinder[0, 0][0]
        if colour == 0:
            hsv_boundries = [np.array([target_hue - 20, 150, 70]), np.array([target_hue + 20, 255, 255])]
        elif colour == 1:
            hsv_boundries = [np.array([target_hue - 0, 70, 70]), np.array([target_hue + 0, 255, 255])]
        else:
            hsv_boundries = [np.array([target_hue - 20, 230, 230]), np.array([target_hue + 20, 255, 255])]

        # Generate the mask based on the desired hue range        
        mask = cv2.inRange(im_hsv, hsv_boundries[0], hsv_boundries[1])
        mask_on_image = cv2.bitwise_and(im_hsv, im_hsv, mask=mask)

        # Debug window to show the mask
        if self.DEBUG:
            cv2.imshow("mask" + str(index), mask_on_image)
            cv2.waitKey(1)

        # Clean up the image
        seg = mask 
        seg = cv2.GaussianBlur(seg, (5, 5), 0)
        seg = cv2.erode(seg, None, iterations=2)
        seg = cv2.dilate(seg, None, iterations=2)        
         
        contours, hierarchy = cv2.findContours(seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)        
        

        if not contours:           
            return

        # Get the largest rectangle
        max_rectangle = None
        self.w = 0
        self.h = 0                
        dst = frame.copy()
        for cnt in contours:
            ## Get the straight bounding rect            
            bbox = cv2.boundingRect(cnt)                 
            x,y,w,h = bbox            

            ## Draw rect
            cv2.rectangle(dst, (x,y), (x+w,y+h), (255,0,0), 1, 16)                           

            ## Get the rotated rect
            rbox = cv2.minAreaRect(cnt)

            if  w*h >= self.w*self.h:
                self.w = w
                self.h = h
                max_rectangle = rbox 
                                 
        # This shouldn't happen, but you never know...
        if max_rectangle is None:                        
            return    

        # Append detected rectangle        
        box = cv2.boxPoints(max_rectangle)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)        
        if self.DEBUG:
            cv2.imshow("rectangles" + str(index), frame)
            cv2.waitKey(1)

        # Normalise values to: x,y = [-0.5, 0.5], r = [0, 1]           
        target_rectangle = np.array(max_rectangle[0]).astype("float32")        
        target_rectangle[0] -= self.x_centre
        target_rectangle[0] /= self.frame_width
        target_rectangle[1] -= self.y_centre
        target_rectangle[1] /= self.frame_width
        target_rectangle[1] *= -1.0       

        # Return a list values [cx, cy, w, h] for the largest rectangle        
        return [target_rectangle[0], target_rectangle[1]]

    def look_for_cylinder(self, colour):
            for index in range(2):  # For each camera (0 = left, 1 = right)
                # Skip if there's no new image, in case the network is choking
                if not self.new_frame[index]:
                    continue
                image = self.input_camera[index]
                # Run the detect cylinder procedure
                self.cylinder[index] = self.detect_cylinder(image, index, colour)            
            # If no cylinder has been detected
            if not self.cylinder[0] and not self.cylinder[1]:
                return False
            else:
                return True
    
#     def detectBlue(self):
#         if (self.detect_cylinder(colour=0)!= NULL):
#             return True
#         else:
#             return False
#     def detectGreen(self):
#         if (self.detect_cylinder(colour=2)!= NULL):
#             return True
#         else:
#             return False
    
    def loop(self):
        print("Starting the loop")
        while not rospy.core.is_shutdown():
            #pick random instruction
            self.instruction = np.random.randint(0, 2)
            print("Instruction: " + str(self.instruction))
            #say instruction until cylinder is moved
            self.frameMissingCountGreen = 0
            self.frameMissingCountBlue = 0
            while self.frameMissingCountGreen<10 and self.frameMissingCountBlue<10:
                if not self.new_frame[0]:
                    rospy.sleep(self.TICK)
                    continue
                #self.beep_pub.publish(self.messages[self.instruction])
                self.greenSeen = self.look_for_cylinder(2)
                rospy.sleep(self.TICK)
                self.blueSeen = self.look_for_cylinder(0)
                #print(self.blueSeen)
                if not (self.greenSeen) :
                    self.frameMissingCountGreen = self.frameMissingCountGreen + 1
                else:
                    self.frameMissingCountGreen = 0
                if not (self.blueSeen) :
                    self.frameMissingCountBlue = self.frameMissingCountBlue + 1
                else:
                    self.frameMissingCountBlue = 0
                rospy.sleep(self.TICK)
            #identify which cylinder is missing
            if (self.frameMissingCountGreen<10):
                self.missing = 1
                print("green missing")
            if (self.frameMissingCountBlue<10):
                self.missing = 2
                print("blue missing")
            #break for student to stop
            start_of_break = rospy.Time.now()
            print("Starting Break")
            while rospy.Time.now() < start_of_break + rospy.Duration(5.0):
                rospy.sleep(self.TICK)
            #decide whether to reward or punish (results)
            if(self.instruction == self.missing):
                self.resultMessage = self.messages[2]
                print("Correct")
            else:
                self.resultMessage = self.messages[3]
                print("Wrong")
            #give results
            start_of_results = rospy.Time.now()
            print("Starting results")
            while rospy.Time.now() < start_of_results + rospy.Duration(5.0):
                #self.beep_pub.publish(self.resultMessage)
                rospy.sleep(self.TICK)
                
            self.instruction = 0
            self.missing = 0
            self.resultMessage = 0
            self.greenSeen = True
            self.blueSeen = True
            self.cylinder = [None, None]
            print("----------------------------------")
            
                
        
        
        
if __name__ == "__main__":

    rospy.init_node("teacher", anonymous=True)
    main = Teacher()
    main.loop()
    # message1 = UInt16MultiArray()
    # message1.data = [1000,255,1]

    # message2 = UInt16MultiArray()
    # message2.data = [1300,255,1]

    # message3 = UInt16MultiArray()
    # message3.data = [1600,255,1]

    # message4 = UInt16MultiArray()
    # message4.data = [2000,255,1]

    # while not rospy.core.is_shutdown():

    #         main.beep_pub.publish(message1)
    #         rospy.sleep(1)
    #         main.beep_pub.publish(message2)
    #         rospy.sleep(1)
    #         main.beep_pub.publish(message3)
    #         rospy.sleep(1)
    #         main.beep_pub.publish(message4)
    #         rospy.sleep(1)
    
