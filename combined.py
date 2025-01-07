# -*- coding: utf-8 -*-
"""
Created on Sun July  1 12:39:06 2024

@author: LapTop
"""


# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:14:24 2024

@author: LapTop
"""



import cv2
import numpy as np

class gaussian:
    def __init__(self):
        self.mean = np.zeros((1, 3))
        self.covariance = 0
        self.weight = 0;
        self.Next = None
        self.Previous = None

class Node:# store info of pixels and num of components
    def __init__(self):
        self.pixel_s = None #root 
        self.pixel_r = None #size last pixel 
        self.no_of_components = 0 # number of components assiotaed with the node object 
        self.Next = None

class Node1: # to create Linked List of gaussian of particular pixels
    def __init__(self):
        self.gauss = None
        self.no_of_comp = 0
        self.Next = None

covariance0 = np.eye(3)
def Create_gaussian(info1, info2, info3):
    ptr = gaussian() #create an object from the guasian class and sotre it in the variable ptr 
    if ptr is not None:
        # intailize the guassian means 
        ptr.mean[0, 0] = info1
        ptr.mean[0, 1] = info2
        ptr.mean[0, 2] = info3
        ptr.covariance = covariance0
        # Initialize weight with a uniform distribution
        ptr.weight = np.random.uniform()
        ptr.Next = None
        ptr.Previous = None #The Next and Previous properties are initialized to None, indicating that this object is the first and only object in the linked list.

    return ptr


    return ptr


    return ptr

def Create_Node(info1, info2, info3):
    N_ptr = Node()
    if (N_ptr is not None):
        N_ptr.Next = None #
        N_ptr.no_of_components = 1
        N_ptr.pixel_s = N_ptr.pixel_r = Create_gaussian(info1, info2, info3)

    return N_ptr

List_node = []
def Insert_End_Node(n):
    List_node.append(n)

List_gaussian = []
def Insert_End_gaussian(n):
    List_gaussian.append(n)

def Delete_gaussian(n):
    List_gaussian.remove(n);

class Process:
    def __init__(self, alpha, history_size):
        self.alpha = alpha
        self.history_size = history_size
        self.background_history = []  # List to store previous frames
        self.background_model = None  # Current background model

    def update_background_model(self, frame):
        # Update background model using the history of frames
        if len(self.background_history) < self.history_size:
            self.background_history.append(frame.copy())
        else:
            self.background_history.pop(0)
            self.background_history.append(frame.copy())
        
        # Update background model with history frames
        background_model = self.background_history[0].copy()
        for hist_frame in self.background_history[1:]:
            background_model = hist_frame * self.alpha + background_model * (1 - self.alpha)

        self.background_model = background_model.astype(np.uint8)

    def get_value(self, frame):
        if self.background_model is None:
            self.background_model = frame.copy()  # Initialize background model
        
        # Update background model
        self.update_background_model(frame)

        # Calculate the absolute difference between the frame and background model
        diff = cv2.absdiff(self.background_model, frame)
        return diff


def denoise(frame):
    width = 640
    height = 360
    aspect_ratio = frame.shape[1] / frame.shape[0]
    if aspect_ratio > 1:
        new_width = width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = height
        new_width = int(new_height * aspect_ratio)
    
    frame = cv2.resize(frame, (new_width, new_height))
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


# Open webcam
capture = cv2.VideoCapture(r"C:\motion\videos\Walking (8).mp4")
frame_counter = 0
ret, orig_frame = capture.read()

# Check if webcam is opened successfully
if ret:
    orig_frame = denoise(orig_frame)
    value1 = Process(0.5, 7)
    run = True
else:
    run = False

# Main loop
while run:
    # Read frame from webcam
    ret, frame = capture.read()
    value = True
    
    # Check if frame is read successfully
    if ret:
        # Apply denoising
        cv2.imshow('input', denoise(frame))
        
        # Process frame
        grayscale = value1.get_value(denoise(frame))
        
        # Apply thresholding
        ret, mask = cv2.threshold(grayscale, 10, 255, cv2.THRESH_BINARY)
        
        # Apply dilation and erosion to the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # Display mask
        cv2.imshow('mask', mask)
# Save the binary mask frame to the specified directory as PNG formatGA
        save_path = r'C:\motion\New folder\mask_frame_{}.png'.format(frame_counter)
        cv2.imwrite(save_path, mask)
        frame_counter += 1
        # Wait for key press
        key = cv2.waitKey(10) & 0xFF
    else:
        break
    

    # Check for ESC key press
    if key == 27:
        break

# Release resources and close windows
capture.release()
cv2.destroyAllWindows()




if value:  # Simplified the condition, as 'value' is already a boolean
        orig_frame = cv2.resize(orig_frame, (340, 260), interpolation=cv2.INTER_CUBIC)
        orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        orig_image_row, orig_image_col = orig_frame.shape  # Using shape attribute for dimensions
    
        bin_frame = np.zeros((orig_image_row, orig_image_col))
    
        for i in range(orig_image_row):
            for j in range(orig_image_col):
                pixel_info = orig_frame[i, j]  # Extracting pixel info directly
                N_ptr = Create_Node(*pixel_info)  # Assuming Create_Node takes (info1, info2, info3)
                if N_ptr is not None:
                    N_ptr.pixel_s.weight = 1.0
                    Insert_End_Node(N_ptr)
                else:
                    raise ValueError("Error creating node")  # Raise exception instead of exiting
    
        nL, nC = orig_image_row, orig_image_col
    
        # Rest of the code...


        dell = np.array((1, 3));
        mal_dist = 0.0;
        temp_cov = 0.0;
        alpha = 0.002;
        cT = 0.05;
        cf = 0.1;
        cfbar = 1.0 - cf;
        alpha_bar = 1.0 - alpha;
        prune = -alpha * cT;
        cthr = 0.00001;
        var = 0.0
        muG = 0.0;
        muR = 0.0;
        muB = 0.0;
        dR = 0.0;
        dB = 0.0;
        dG = 0.0;
        rval = 0.0;
        gval = 0.0;
        bval = 0.0;

        while (1):
            duration3 = 0.0;
            count = 0;
            count1 = 0;
            List_node1 = List_node;
            counter = 0;
            duration = cv2.getTickCount( );
            for i in range(0, nL):
                r_ptr = orig_frame[i]
                b_ptr = bin_frame[i]

                for j in range(0, nC):
                    sum = 0.0;
                    sum1 = 0.0;
                    close = False;
                    background = 0;

                    rval = r_ptr[0][0];
                    gval = r_ptr[0][0];
                    bval = r_ptr[0][0];

                    start = List_node1[counter].pixel_s;
                    rear = List_node1[counter].pixel_r;
                    ptr = start;

                    temp_ptr = None;
                    if (List_node1[counter].no_of_component > 4):
                        Delete_gaussian(rear);
                        List_node1[counter].no_of_component = List_node1[counter].no_of_component - 1;

                    for k in range(0, List_node1[counter].no_of_component):
                        weight = List_node1[counter].weight;
                        mult = alpha / weight;
                        weight = weight * alpha_bar + prune;
                        if (close == False):
                            muR = ptr.mean[0];
                            muG = ptr.mean[1];
                            muB = ptr.mean[2];

                            dR = rval - muR;
                            dG = gval - muG;
                            dB = bval - muB;

                            var = ptr.covariance;

                            mal_dist = (dR * dR + dG * dG + dB * dB);

                            if ((sum < cfbar) and (mal_dist < 16.0 * var * var)):
                                background = 255;

                            if (mal_dist < (9.0 * var * var)):
                                weight = weight + alpha;
                                if mult < 20.0 * alpha:
                                    mult = mult;
                                else:
                                    mult = 20.0 * alpha;

                                close = True;

                                ptr.mean[0] = muR + mult * dR;
                                ptr.mean[1] = muG + mult * dG;
                                ptr.mean[2] = muB + mult * dB;
                                temp_cov = var + mult * (mal_dist - var);
                                if temp_cov < 5.0:
                                    ptr.covariance = 5.0
                                else:
                                    if (temp_cov > 20.0):
                                        ptr.covariance = 20.0
                                    else:
                                        ptr.covariance = temp_cov;

                                temp_ptr = ptr;

                        if (weight < -prune):
                            ptr = Delete_gaussian(ptr);
                            weight = 0;
                            List_node1[counter].no_of_component = List_node1[counter].no_of_component - 1;
                        else:
                            sum += weight;
                            ptr.weight = weight;

                        ptr = ptr.Next;

                    if (close == False):
                        ptr = gaussian( );
                        ptr.weight = alpha;
                        ptr.mean[0] = rval;
                        ptr.mean[1] = gval;
                        ptr.mean[2] = bval;
                        ptr.covariance = covariance0;
                        ptr.Next = None;
                        ptr.Previous = None;
                        Insert_End_gaussian(ptr);
                        List_gaussian.append(ptr);
                        temp_ptr = ptr;
                        List_node1[counter].no_of_components = List_node1[counter].no_of_components + 1;

                    ptr = start;
                    while (ptr != None):
                        ptr.weight = ptr.weight / sum;
                        ptr = ptr.Next;

                    while (temp_ptr != None and temp_ptr.Previous != None):
                        if (temp_ptr.weight <= temp_ptr.Previous.weight):
                            break;
                        else:
                            next = temp_ptr.Next;
                            previous = temp_ptr.Previous;
                            if (start == previous):
                                start = temp_ptr;
                                previous.Next = next;
                                temp_ptr.Previous = previous.Previous;
                                temp_ptr.Next = previous;
                            if (previous.Previous != None):
                                previous.Previous.Next = temp_ptr;
                            if (next != None):
                                next.Previous = previous;
                            else:
                                rear = previous;
                                previous.Previous = temp_ptr;

                        temp_ptr = temp_ptr.Previous;

                    List_node1[counter].pixel_s = start;
                    List_node1[counter].pixel_r = rear;
                    counter = counter + 1;


capture.release()
cv2.destroyAllWindows()
