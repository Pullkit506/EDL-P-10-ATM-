import tkinter as tk
import cv2
from picamera2 import Picamera2
import time
import iris
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import socket

def ir_image(img, red_weight = 2.1, green_weight = 0.8, blue_weight = 1):

    # Split the image into red, green, and blue channels
    b, g, r = cv2.split(img)

    # Apply custom weights to each channel
    b = np.uint8(np.clip(b * blue_weight, 0, 255))
    g = np.uint8(np.clip(g * green_weight, 0, 255))
    r = np.uint8(np.clip(r * red_weight, 0, 255))

    # Merge the channels back together
    result = cv2.merge([b, g, r])

    # Convert the image to grayscale
    bw_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # bw_image = cv2.equalizeHist(bw_image)
    return bw_image

def calculate_masked_fractional_hd(iris_code1, iris_code2, mask1, mask2):
        flat_code1 = iris_code1.flatten()
        flat_code2 = iris_code2.flatten()

        xor = flat_code1 ^ flat_code2
        mask = mask1.flatten() & mask2.flatten()

        masked_code = xor & mask
        hamming_distance = np.sum(masked_code)
        frac = np.sum(mask)
        masked_fractional_hd = hamming_distance / frac

        return masked_fractional_hd

class IrisVerificationApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Iris Verification App")

        self.picam2 = Picamera2()
        camera_config = self.picam2.create_still_configuration(main={"size": (3280, 2464)}, lores={"size": (640, 480)}, display="lores")
        self.picam2.configure(camera_config)
        self.picam2.start()
        time.sleep(1)

        # self.picam2 = Picamera2()
        # self.camera_config = self.picam2.create_still_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="lores")
        # self.picam2.configure(self.camera_config)
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.result_label = tk.Label(self.root, text="Iris Classifier Model", font=("Helvetica", 16))
        self.result_label.pack(pady=20)

        self.capture_button = tk.Button(self.root, text="Capture Iris", command=self.capture_image)
        self.capture_button.pack(pady=10)

        self.verify_button = tk.Button(self.root, text="Verify Iris", command=self.verify_iris, state=tk.DISABLED)
        self.verify_button.pack(pady=10)

        self.add_to_database_button = tk.Button(self.root, text="Add to Database", command=self.add_to_database, state=tk.DISABLED)
        self.add_to_database_button.pack(pady=10)

        self.iris_pipeline = iris.IRISPipeline(env=iris.IRISPipeline.DEBUGGING_ENVIRONMENT)
        self.database = {}

        self.capture_running = 1
        self.output = {}


    def capture_image(self):
        if(self.capture_running==1):
            img = self.picam2.capture_array("main")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, minSize=(1000, 1000))
            
            if len(eyes) != 0:
                self.result_label.config(text="Eye Detected. Stand Still for 2 second")
                self.result_label.update_idletasks() 
                self.root.after(1000, self.continue_processing)
            else:
                self.result_label.config(text="Detecting")
                self.result_label.update_idletasks() 
                self.root.after(33, self.capture_image)

    def continue_processing(self):
        img = self.picam2.capture_array("main")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, minSize=(1000, 1000))

        if(len(eyes)==0):
            self.result_label.config(text="Try Again")
            self.result_label.update_idletasks() 
            self.capture_image()
        else:
            self.result_label.config(text="Image Captured.")
            self.result_label.update_idletasks() 

            largest_index = max(range(len(eyes)), key=lambda i: eyes[i][2] * eyes[i][3])
            (x, y, w, h) = eyes[largest_index]

            eye_img = img[y:y + h, x:x + w]
            # rgb_frame = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)

            cv2.imwrite('saved_frame_complete.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.imwrite('saved_frame_rgb.jpg', cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
            cv2.imwrite('saved_frame_ir.jpg', ir_image(eye_img))

            self.output = self.iris_pipeline(img_data=ir_image(eye_img), eye_side="right")

            if self.output['error'] is None:
                self.result_label.config(text="Iris detected. Click 'Verify Iris' or 'Add to database' to continue.")
                self.verify_button.config(state=tk.NORMAL)
                self.add_to_database_button.config(state=tk.NORMAL)
                self.result_label.update_idletasks() 
                self.capture_running = 0
                self.root.after(2000)
            else:
                self.result_label.config(text="Iris Not detected")
                self.result_label.update_idletasks() 
                self.root.after(2000, self.capture_image)
            
    def verify_iris(self):
        
        for user_id in self.database:
            database_template = self.database[user_id]
            hd = calculate_masked_fractional_hd(self.output['iris_template']['iris_codes'], database_template['iris_codes'],
                                                self.output['iris_template']['mask_codes'], database_template['mask_codes'])
            if hd < 0.4:
                self.result_label.config(text=f"Verification result for user {user_id}: Passed and {hd}")
                self.result_label.update_idletasks() 
                break
        else:
            self.result_label.config(text=f"Verification Failed and {hd}")
            self.result_label.update_idletasks() 

        self.capture_running = 1
        self.root.after(2000, self.capture_image())

    def add_to_database(self):
        user_id = len(self.database) + 1
        self.database[user_id] = self.output['iris_template']
        self.result_label.config(text=f"User {user_id} added to the database.")
        self.result_label.update_idletasks() 
        self.capture_running = 1
        self.root.after(2000, self.capture_image())

if _name_ == "_main_":
    root = tk.Tk()
    app = IrisVerificationApp(root)
    root.mainloop()
