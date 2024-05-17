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


class class1:

    def load_image(self):
        image = Image.open(self.get_image())
        self.apply_Edit(image)

    def get_image(self):
        return "baboon.png"

    def apply_Edit(self, edited_image):
        # Converting
        edited_photo = ImageTk.PhotoImage(edited_image)
        # Update the label
        image_label.config(image=edited_photo)
        # Keep a reference to the image to prevent garbage collection
        image_label.image = edited_photo

    def Add_Button( self,func , button_name):
        button = Button(window, text=f"{button_name}")
        button.config(command=func)
        button.config(font=("Helvetica", 20,))
        button.pack()

    def sobel_edge_detector(self):
        img_gray = cv2.imread(self.get_image(), cv2.IMREAD_GRAYSCALE)

        # filter the image to get ride of noise caused by world
        # img_gray = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0.1, sigmaY=0.1)

        ddepth = cv2.CV_16S

        # Applying the filter on the image in the X direction
        grad_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3)

        # Applying the filter on the image in the Y direction
        grad_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
        # Converts the values back to a number between 0 and 255
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # Adds the derivative in the X and Y direction
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # convert the grad image to a PIL image
        pil_image = Image.fromarray(np.uint8(grad))

        self.apply_Edit(pil_image)

    def prewitt_edge_detector(self):
        # Convert the original image to grayscale

        gray_image = cv2.imread(self.get_image(), cv2.IMREAD_GRAYSCALE)

        # Compute the horizontal and vertical gradients using Prewitt operators
        prewitt_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        prewitt_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute the magnitude of gradients
        prewitt_image = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
        # Normalize the gradient magnitude image
        prewitt_image = cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Update the image display with the Prewitt edge detected image
        pil_image = Image.fromarray(prewitt_image)
        self.apply_Edit(pil_image)



    def roberts_edge_detector(self):
        image = Image.open(self.get_image())

        gray_image = np.array(image)

        filtered_image = cv2.Canny(gray_image, 100, 200)

        pil_Image = Image.fromarray(filtered_image)
        self.apply_Edit(pil_Image)

    def Buttons(self):

        # Adding the button to load original image
        self.Add_Button(self.load_image, "load image")

        # Adding the button of roberts_edge_detector filter
        self.Add_Button(self.roberts_edge_detector,"roberts edge detector")

        # Adding the button of mean filter
        self.Add_Button(self.prewitt_edge_detector, "prewitt edge detector")

        # Adding the button of sobel edge detector filter
        self.Add_Button(self.sobel_edge_detector, "sobel edge detector")

    ###################################################################################

    # Crating a window


window = Tk()
window.geometry("700x2000")
window.configure(bg="darkslategrey")
# window.resizable(0,0)

# Adding Image to the window
image = Image.open("baboon.png")
image = ImageTk.PhotoImage(image)
image_label = Label(window, image=image)
image_label.pack()
image_label.pack()

execute = class1()
execute.Buttons()

window.mainloop()
