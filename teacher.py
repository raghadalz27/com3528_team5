import os
import numpy as np
import rospy
import miro2 as miro
from std_msgs.msg import UInt16MultiArray, Int16MultiArray
import cv2
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message

class Teacher():
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
        self.cylinder_one = [None, None]
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
        message1 = UInt16MultiArray()
        message1.data = [1000,255,1]
        message2 = UInt16MultiArray()
        message2.data = [1300,255,1]
        message3 = UInt16MultiArray()
        message3.data = [1600,255,1]
        message4 = UInt16MultiArray()
        message4.data = [2000,255,1]
        self.messages = [message1,message2,message3,message4]
        self.greenSeen = True
        self.blueSeen = True
        self.frameMissingCount = 0
        self.DEBUG = True

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
            self.input_camera[index] = image
            # Get image dimensions
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2.0
            self.y_centre = self.frame_height / 2.0
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
            hsv_boundries = [np.array([target_hue - 20, 70, 70]), np.array([target_hue + 20, 255, 255])]
        elif colour == 1:
            hsv_boundries = [np.array([target_hue - 0, 70, 70]), np.array([target_hue + 0, 255, 255])]
        else:
            hsv_boundries = [np.array([target_hue - 20, 70, 70]), np.array([target_hue + 20, 255, 255])]

        # Generate the mask based on the desired hue range
        ##NOTE Both masks are currently blue
        mask = cv2.inRange(im_hsv, hsv_boundries[0], hsv_boundries[1])
        mask_on_image = cv2.bitwise_and(im_hsv, im_hsv, mask=mask)

        # Debug window to show the mask
        if self.DEBUG:
            cv2.imshow("mask" + str(index), mask_on_image)
            cv2.waitKey(1)

        # Clean up the image
        seg = mask # Currently only looks for blue
        seg = cv2.GaussianBlur(seg, (5, 5), 0)
        seg = cv2.erode(seg, None, iterations=2)
        seg = cv2.dilate(seg, None, iterations=2)

        # Fine-tune parameters
        cylinder_detect_min_dist_between_cens = 40  # Empirical
        canny_high_thresh = 10  # Empirical
        cylinder_detect_sensitivity = 10  # Lower detects more circles, so it's a trade-off
        cylinder_detect_min_radius = 5  # Arbitrary, empirical
        cylinder_detect_max_radius = 50  # Arbitrary, empirical
         
        ##NOTE Need to change to find cylinder boundaries using hough line transform
        # Find circles using OpenCV routine
        # This function returns a list of circles, with their x, y and r values
        circles = cv2.HoughCircles(
            seg,
            cv2.HOUGH_GRADIENT,
            1,
            cylinder_detect_min_dist_between_cens,
            param1=canny_high_thresh,
            param2=cylinder_detect_sensitivity,
            minRadius=cylinder_detect_min_radius,
            maxRadius=cylinder_detect_max_radius,
        )

        if circles is None:
            # If no circles were found, just display the original image
            return

        # Get the largest circle
        max_circle = None
        self.max_rad = 0
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            if c[2] > self.max_rad:
                self.max_rad = c[2]
                max_circle = c
        # This shouldn't happen, but you never know...
        if max_circle is None:
            return

        # Append detected circle and its centre to the frame
        cv2.circle(frame, (max_circle[0], max_circle[1]), max_circle[2], (0, 255, 0), 2)
        cv2.circle(frame, (max_circle[0], max_circle[1]), 2, (0, 0, 255), 3)
        if self.DEBUG:
            cv2.imshow("circles" + str(index), frame)
            cv2.waitKey(1)

        # Normalise values to: x,y = [-0.5, 0.5], r = [0, 1]
        max_circle = np.array(max_circle).astype("float32")
        max_circle[0] -= self.x_centre
        max_circle[0] /= self.frame_width
        max_circle[1] -= self.y_centre
        max_circle[1] /= self.frame_width
        max_circle[1] *= -1.0
        max_circle[2] /= self.frame_width

        # Return a list values [x, y, r] for the largest circle
        return [max_circle[0], max_circle[1], max_circle[2]]
    
    def detectBlue(self):
        if (detect_cylinder(colour=0)!= NULL):
            return True
        else:
            return False
    def detectGreen(self):
        if (detect_cylinder(colour=2)!= NULL):
            return True
        else:
            return False
        
        
    def loop(self):
        print("Starting the loop")
        while not rospy.core.is_shutdown():
            #pick random instruction
            self.instruction = np.random.randint(0, 2)
            print("Instruction: " + str(self.instruction))
            #say instruction until cylinder is moved
            self.frameMissingCount = 0
            while self.frameMissingCount<4:
                #self.beep_pub(self.messages[self.instruction])
                self.greenSeen = self.detectGreen()
                self.blueSeen = self.detectBlue()
                if not (self.greenSeen == True and self.blueSeen == True) :
                    self.frameMissingCount = self.frameMissingCount + 1
            #identify which cylinder is missing
            if(self.blueSeen == False):
                self.missing = 1
                print("blue missing")
            else:
                self.missing = 2
                print("green missing")
            #break for student to stop
            start_of_break = rospy.Time.now()
            print("Starting Break")
            while rospy.Time.now() < start_of_break + rospy.Duration(3.0):
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
                #self.beep_pub(self.resultMessage)
                rospy.sleep(self.TICK)
                
            self.instruction = 0
            self.missing = 0
            self.resultMessage = 0
            self.greenSeen = True
            self.blueSeen = True
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
    
