from tkinter import *
from tkinter import ttk
from scipy import ndimage
import cv2
import os
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
import tkinter as tk
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

class class3:

    def load_image(self):
        image = Image.open(self.get_image())
        self.apply_Edit(image)

    def get_image(self):
        return "balls.jpg"

    def apply_Edit(self,edited_image):
        # Converting
        edited_photo = ImageTk.PhotoImage(edited_image)
        # Update the label
        image_label.config(image=edited_photo)
        # Keep a reference to the image to prevent garbage collection
        image_label.image = edited_photo

    def thresholding(self):
        gray_image = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)
        threshold_value = 127

        _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
        pil_image = Image.fromarray(thresholded_image)
        self.apply_Edit(pil_image)

    def Add_Button(self,func, button_name):
        button = Button(window, text=f"{button_name}")
        button.config(command=func)
        button.config(font=("Helvetica", 25, ))
        button.pack()

    def hough_circle_transform(self):
        # Load the image
        original_image = cv2.imread(self.get_image())

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Detect circles using Hough circle transform with specified parameters
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30,
                                   minRadius=0,
                                   maxRadius=0)

        # Check if any circles are detected
        if circles is not None:
            # Convert the circle parameters to integer
            circles = np.uint16(np.around(circles))

            # Draw detected circles on the original image
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                # Draw the outer circle
                cv2.circle(original_image, center, radius, (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(original_image, center, 2, (0, 0, 255), 3)

            # Convert the image to PIL format and update the display
            pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            self.apply_Edit(pil_image)

    def Buttons(self):
        self.Add_Button(self.load_image,'load Image')
        self.Add_Button(self.thresholding, "Thresholding")
        self.Add_Button(self.hough_circle_transform, "hough circle transform")










window = Tk()
window.geometry("700x2000")
window.configure(bg="darkslategrey")
# window.resizable(0,0)

# Adding Image to the window
image = Image.open("balls.jpg")
image = ImageTk.PhotoImage(image)
image_label = Label(window, image=image)
image_label.pack()
image_label.pack()

execute = class3()
execute.Buttons()

window.mainloop()