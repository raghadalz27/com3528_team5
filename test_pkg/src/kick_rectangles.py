#!/usr/bin/env python3

import os
from math import radians
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from sensor_msgs.msg import JointState  # ROS joints state message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control) message

import miro2 as miro  # Import MiRo Developer Kit library

try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2

class MiRoClient:
    TICK = 0.02  # This is the update interval for the main control loop in secs
    CAM_FREQ = 1  # Number of ticks before camera gets a new frame, increase in case of network lag
    SLOW = 0.1  # Radial speed when turning on the spot (rad/s)
    FAST = 0.8  # Linear speed when kicking the cylinder (m/s)
    DEBUG = True  # Set to True to enable debug views of the cameras
    COLOUR = 0 # 0 = BLUE, 1 = RED, 2 = GREEN

    ##NOTE The following option is relevant in MiRoCODE
    NODE_EXISTS = False  # Disables (True) / Enables (False) rospy.init_node

    def reset_head_pose(self):
        """
        Reset MiRo head to default position, to avoid having to deal with tilted frames
        """
        self.kin_joints = JointState()  # Prepare the empty message
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, radians(35.0), 0.0, 0.0]
        t = 0
        while not rospy.core.is_shutdown():  # Check ROS is running
            # Publish state to neck servos for 1 sec
            self.pub_kin.publish(self.kin_joints)
            rospy.sleep(self.TICK)
            t += self.TICK
            if t > 1:
                break

    def drive(self, speed_l=0.1, speed_r=0.1):  # (m/sec, m/sec)
        """
        Wrapper to simplify driving MiRo by converting wheel speeds to cmd_vel
        """
        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Desired wheel speed (m/sec)
        wheel_speed = [speed_l, speed_r]

        # Convert wheel speed to command velocity (m/sec, Rad/sec)
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        # Update the message with the desired speed
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        # Publish message to control/cmd_vel topic
        self.vel_pub.publish(msg_cmd_vel)

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
            # Crop image
            #image = image[80:280, 150:300]
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

    def detect_cylinder(self, frame, index):
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
        hsv_cylinder = cv2.cvtColor(rgb_cylinder[self.COLOUR], cv2.COLOR_RGB2HSV)        

        # Extract colour boundaries for masking image
        # Get the hue value from the numpy array containing target colour
        target_hue = hsv_cylinder[0, 0][0]
        if self.COLOUR == 0:
            hsv_boundries = [np.array([target_hue - 20, 70, 70]), np.array([target_hue + 20, 255, 255])]
        elif self.COLOUR == 1:
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
         
        # Find rectangles        
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # h,s,v = cv2.split(hsv)
        # h[h<150]=0
        # h[h>180]=0

        ## normalize, do the open-morp-op
        # normed = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
        # opened = cv2.morphologyEx(normed, cv2.MORPH_OPEN, kernel)            

        contours, hierarchy = cv2.findContours(seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)        
        
        #print(contours)

        if not contours:
            # print("No contours")
            return

        # Get the largest rectangle
        max_rectangle = None
        self.w = 0
        self.h = 0                
        dst = frame.copy()
        for cnt in contours:
            ## Get the straight bounding rect            
            bbox = cv2.boundingRect(cnt)   
            #print(bbox)     
            # if bbox is None:
            #     print("bbox is none")    
            x,y,w,h = bbox
            # if w < 10 or h < 10 or w*h < 1500 or w > 1000:
            #     continue

            ## Draw rect
            cv2.rectangle(dst, (x,y), (x+w,y+h), (255,0,0), 1, 16)                           

            ## Get the rotated rect
            rbox = cv2.minAreaRect(cnt)
            #(cx,cy), (w,h), rot_angle = rbox       

            if  w*h >= self.w*self.h:
                self.w = w
                self.h = h
                max_rectangle = rbox 
                # print(f"self: ({self.w}, {self.h}), w:{w}, h:{h}")
                # print(f"bbox: {bbox}")
            # else:
            #     print("rectangle is none")                  
        # This shouldn't happen, but you never know...
        if max_rectangle is None:  
            #print("max rectangle is none")          
            return    

        # Append detected rectangle        
        box = cv2.boxPoints(max_rectangle)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)
        #cv2.rectangle(frame, (max_rectangle[0], max_rectangle[1]), (max_rectangle[0]+max_rectangle[2], max_rectangle[1]+max_rectangle[3]), (0, 255, 0), 2)        
        if self.DEBUG:
            cv2.imshow("rectangles" + str(index), frame)
            cv2.waitKey(1)

        # Normalise values to: x,y = [-0.5, 0.5], r = [0, 1]
        #print(max_rectangle)        
        #max_rectangle = np.array(max_rectangle).astype("float32")       
        target_rectangle = np.array(max_rectangle[0]).astype("float32")
        # max_rectangle[0] = ((max_rectangle[0] + max_rectangle[2]) / 2) - self.x_centre
        # max_rectangle[0] = ((max_rectangle[0] + max_rectangle[2]) / 2) / self.frame_width
        # max_rectangle[1] = ((max_rectangle[1] + max_rectangle[3]) / 2) - self.y_centre
        # max_rectangle[1] = ((max_rectangle[1] + max_rectangle[3]) / 2) / self.frame_width
        target_rectangle[0] -= self.x_centre
        target_rectangle[0] /= self.frame_width
        target_rectangle[1] -= self.y_centre
        target_rectangle[1] /= self.frame_width
        target_rectangle[1] *= -1.0       

        # Return a list values [cx, cy, w, h] for the largest rectangle
        #return [max_rectangle[0], max_rectangle[1], max_rectangle[2], max_rectangle[3]]
        return [target_rectangle[0], target_rectangle[1]]

    def look_for_cylinder(self):
        """
        [1 of 3] Rotate MiRo if it doesn't see a cylinder in its current
        position, until it sees one.
        """
        if self.just_switched:  # Print once
            print("MiRo is looking for the cylinder...")
            self.just_switched = False
        for index in range(2):  # For each camera (0 = left, 1 = right)
            # Skip if there's no new image, in case the network is choking
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect cylinder procedure
            self.cylinder[index] = self.detect_cylinder(image, index)

        # If no cylinder has been detected
        if not self.cylinder[0] and not self.cylinder[1]:
            self.drive(self.SLOW, -self.SLOW)
        else:
            self.status_code = 2  # Switch to the second action
            self.just_switched = True

    def lock_onto_cylinder(self, error=25):
        """
        [2 of 3] Once a cylinder has been detected, turn MiRo to face it
        """
        if self.just_switched:  # Print once
            print("MiRo is locking on to the cylinder")
            self.just_switched = False
        for index in range(2):  # For each camera (0 = left, 1 = right)
            # Skip if there's no new image, in case the network is choking
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect cylinder procedure
            self.cylinder[index] = self.detect_cylinder(image, index)

        # If only the right camera sees the cylinder, rotate clockwise
        if not self.cylinder[0] and self.cylinder[1]:
            #print(f"turning right")
            self.drive(self.SLOW, -self.SLOW)
        # Conversely, rotate counterclockwise
        elif self.cylinder[0] and not self.cylinder[1]:
            #print(f"turning left, 0: {self.cylinder[0]}, 1: {self.cylinder[1]}")
            self.drive(-self.SLOW, self.SLOW)
        # Make the MiRo face the cylinder if it's visible with both cameras
        elif self.cylinder[0] and self.cylinder[1]:
            error = 0.08  # 5% of image width
            # Use the normalised values
            left_x = self.cylinder[0][0]  # should be in range [0.0, 0.5]
            right_x = self.cylinder[1][0] # should be in range [-0.5, 0.0]
            rotation_speed = 0.03  # Turn even slower now
            #print(f"left_x={left_x}, right_x={right_x}")
            if abs(left_x) - abs(right_x) > error:
                #print(f"Turning Right: {abs(left_x) - abs(right_x)}")
                self.drive(rotation_speed, -rotation_speed)  # turn clockwise               
            elif abs(left_x) - abs(right_x) < -error:
                #print(f"Turning Left: left_x={left_x}, right_x={right_x}")
                self.drive(-rotation_speed, rotation_speed)  # turn counterclockwise
            else:
                # Successfully turned to face the cylinder
                #print("found it")
                self.status_code = 3  # Switch to the third action
                self.just_switched = True
                self.bookmark = self.counter
        # Otherwise, the cylinder is lost :-(
        else:
            self.status_code = 0  # Go back to square 1...
            print("MiRo has lost the cylinder...")
            self.just_switched = True

    # GOAAAL
    def kick(self):
        """
        [3 of 3] Once MiRO is in position, this function should drive the MiRo
        forward until it kicks the cylinder!
        """
        if self.just_switched:
            print("MiRo is kicking the cylinder!")
            self.just_switched = False
        if self.counter <= self.bookmark + 2 / self.TICK:
            self.drive(self.FAST, self.FAST)
        else:
            self.status_code = 0  # Back to the default state after the kick
            self.just_switched = True

    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo
        if not self.NODE_EXISTS:
            rospy.init_node("kick_blue_cylinder", anonymous=True)
        # Give it some time to make sure everything is initialised
        rospy.sleep(2.0)
        # Initialise CV Bridge
        self.image_converter = CvBridge()
        # Individual robot name acts as ROS topic prefix
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        # Create two new subscribers to receive camera images with attached callbacks
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
        # Create a new publisher to send velocity commands to the robot
        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        # Create a new publisher to move the robot head
        self.pub_kin = rospy.Publisher(
            topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )
        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of cylinder's x, y, and r values for each camera
        self.cylinder = [None, None, None, None]
        # Set the default frame width (gets updated on receiving an image)
        self.frame_width = 640
        # Action selector to reduce duplicate printing to the terminal
        self.just_switched = True
        # Bookmark
        self.bookmark = 0
        # Move the head to default pose
        self.reset_head_pose()

    def loop(self):
        """
        Main control loop
        """
        print("MiRo plays cylinder, press CTRL+C to halt...")
        # Main control loop iteration counter
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find cylinder, lock on to the cylinder and kick cylinder
        self.status_code = 0
        while not rospy.core.is_shutdown():

            # Step 1. Find cylinder
            if self.status_code == 1:
                # Every once in a while, look for cylinder
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_cylinder()

            # Step 2. Orient towards it
            elif self.status_code == 2:
                self.lock_onto_cylinder()

            # Step 3. Kick!
            elif self.status_code == 3:
                self.kick()

            # Fall back
            else:
                self.status_code = 1

            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)


# This condition fires when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop