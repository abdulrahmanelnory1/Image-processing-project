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


class class2:
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

    def Add_Button(self, func, button_name):
        button = Button(window, text=f"{button_name}")
        button.config(command=func)
        button.config(font=("Helvetica", 11,))
        button.pack()

    def highpass_filter(self):
        image = cv2.imread(self.get_image(), cv2.IMREAD_GRAYSCALE)

        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2

        # Define a mask to perform high-pass filtering
        mask = np.ones((rows, cols, 2), np.uint8)
        mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 0

        # Apply the mask to the shifted Fourier transform
        dft_shift = dft_shift * mask

        # Inverse Fourier transform to get the filtered image
        filtered_image = np.fft.ifftshift(dft_shift)
        filtered_image = cv2.idft(filtered_image)
        filtered_image = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])

        # Normalize the filtered image
        filtered_image = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_8U)

        # Convert the filtered image to a PIL image
        pil_image = Image.fromarray(filtered_image)

        # Apply the filtered image to the GUI
        self.apply_Edit(pil_image)

    def lowpass_filter(self):
        image = cv2.imread(self.get_image(), cv2.IMREAD_GRAYSCALE)

        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        pil_image = Image.fromarray(np.uint8(magnitude_spectrum))
        self.apply_Edit(pil_image)


    def mean_filter(self):
        # Load the image
        image = cv2.imread(self.get_image())

        # Define the kernel
        mean_image = cv2.blur(image, (7, 7))
        mean_image = cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(mean_image)
        self.apply_Edit(pil_image)

    def median_filter(self):
        # Read the image in Gray Scale
        image = cv2.imread(self.get_image())

        # Apply the median filter
        filtered_image = cv2.medianBlur(image, 13)

        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

        # convert the grad image to a PIL image
        pil_image = Image.fromarray(np.uint8(filtered_image))

        self.apply_Edit(pil_image)

    def erosion(self):
        # Load the image
        image = cv2.imread(self.get_image())

        # Split the image into its color channels
        b, g, r = cv2.split(image)

        # Get the selected kernel size
        kernel_size = 5
        # Create a kernel for erosion
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Perform erosion on each color channel
        b_filtered = cv2.erode(b, kernel, iterations=1)
        g_filtered = cv2.erode(g, kernel, iterations=1)
        r_filtered = cv2.erode(r, kernel, iterations=1)

        # Merge the filtered color channels back into a BGR image
        filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

        # Convert the filtered image to a PIL image
        pil_image = Image.fromarray(filtered_image)

        # Apply the filtered image to the GUI
        self.apply_Edit(pil_image)

    def open(self):
        # Load the image using OpenCV
        image = cv2.imread(self.get_image())

        # Get the selected kernel size from the slider
        kernel_size = 8
        # Create a kernel for opening
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Perform opening on the original image
        open_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Convert the processed image from BGR to RGB
        open_image = cv2.cvtColor(open_image, cv2.COLOR_BGR2RGB)

        # Convert the processed image to a PIL image
        pil_image = Image.fromarray(open_image)
        # Update the image display with the opened image
        self.apply_Edit(pil_image)

    def close(self):
        image = cv2.imread(self.get_image())

        # Get the selected kernel size from the slider
        kernel_size = 7
        # Create a kernel for closing
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Perform closing on the original image
        close_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        close_image=cv2.cvtColor(close_image, cv2.COLOR_BGR2RGB)
        # Convert the processed image to a PIL image
        pil_image = Image.fromarray(close_image)
        # Update the image display with the opened image
        self.apply_Edit(pil_image)

    def dilation(self):

        image = cv2.imread(self.get_image())

        # Get the selected kernel size from the slider
        kernel_size = 7
        # Create a kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Perform dilation on the original image
        dilation_image = cv2.dilate(image, kernel, iterations=1)
        dilation_image = cv2.cvtColor(dilation_image, cv2.COLOR_BGR2RGB)
        # Convert the processed image to a PIL image
        pil_image = Image.fromarray(dilation_image)
        # Update the image display with the opened image
        self.apply_Edit(pil_image)







    def Buttons(self):
        # Adding the button to load original image
        self.Add_Button(self.load_image, "load image")

        # Adding the button of LPF filter
        self.Add_Button(self.lowpass_filter, "LPF")

        # Adding the button of LPF filter
        self.Add_Button(self.highpass_filter, "HPF")

        # Adding the button of mean filter
        self.Add_Button(self.mean_filter, "mean filter")

        # Adding the button of median filter
        self.Add_Button(self.median_filter, "median filter")

        # Adding the button of median filter
        self.Add_Button(self.erosion, "Erosion")

        # Adding the button of open filter
        self.Add_Button(self.open, "Open")

        # Adding the button of open filter
        self.Add_Button(self.close, "Close")

        # Adding the button of Dilation
        self.Add_Button(self.dilation, "Dilation")


#########################################################################
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

execute = class2()
execute.Buttons()

window.mainloop()
