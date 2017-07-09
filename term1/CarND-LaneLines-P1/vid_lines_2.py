import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
from moviepy.editor import VideoFileClip

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3, consolidate=False):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # If requested, consolidate to two lines and extend    
    if consolidate:
        left, right = ([],[])
        for line in lines:
            for x1,y1,x2,y2 in line:            
                m = (y2-y1)/(x2-x1)
                b = (y1+y2-m*(x1+x2))/2
                # Skip flat lines
                if abs(m)<0.3: continue
                # Left vs. Right
                if m>0:
                    left.append([m,b])
                else:
                    right.append([m,b])

        # Average slope and intersect
        ml,bl = np.mean(left,axis=0)
        mr,br = np.mean(right,axis=0)

        # Limits
        height,width,depth = img.shape
        ymin = int(0.6*height)
        ymax = int(1.0*height)

        # Extend consolidated lines
        line = []
        xmin = int(min(width,max(0,(ymin-bl)/ml)))
        xmax = int(min(width,max(0,(ymax-bl)/ml)))
        line.append([xmin,ymin,xmax,ymax])

        xmin = int(min(width,max(0,(ymin-br)/mr)))
        xmax = int(min(width,max(0,(ymax-br)/mr)))
        line.append([xmin,ymin,xmax,ymax])

        lines= [line]

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return

def hough_lines(img, rho, theta, threshold):
    """
    Computes the hough lines in terms of ρ and θ pairs.
    Consolidates and averages into two lines and returns
    the image of original size with lane lines only
    """

    # Returns (rho,theta) pairs, Using a multiscale hough-transform
    lines  = cv2.HoughLines( img, rho, theta, threshold, np.array([]),1,5).squeeze()

    # Discard Horizontal lines (or close to horizontal)
    discard, left_lane,right_lane = ([],[],[])
    for i in range(lines.shape[0]):
        θ = lines[i,1]
        if abs(θ-np.pi/2) < 0.1:
            # Close to flat. discard
            discard.append(i)
        elif θ < np.pi/2:
            # Left lane line slope < pi/2
            left_lane.append(i)
        else:
            # Right lane line slope > pi/2
            right_lane.append(i)

    # Average rho and theta for both lane lines
    rt_left = np.mean(lines[left_lane,:],axis=0)
    rt_right = np.mean(lines[right_lane,:],axis=0)
    #print("rho,theta (left) = {}\nrho,theta (right) = {}".format(rt_left,rt_right))

    # Obtain (x,y) pairs from (ρ, θ)
    ymin = int(0.6*img.shape[0])
    ymax = int(1.0*img.shape[0])
    lanes = []

    # Function to convert from polar to cartesian coordinates
    # ρ = x.cos(θ) + y.sin(θ) ==> x = (ρ-y.sin(θ))/cos(θ)
    coord_transform = lambda y, ρ, θ : int((ρ-y*np.sin(θ))/np.cos(θ))
    width = img.shape[1]
    ρ, θ = rt_left    
    xmin = min(width,max(0,coord_transform(ymin,ρ,θ)))
    xmax = min(width,max(0,coord_transform(ymax,ρ,θ)))
    lanes.append([[xmin,ymin,xmax,ymax]])

    ρ, θ = rt_right
    xmin = min(width,max(0,coord_transform(ymin,ρ,θ)))
    xmax = min(width,max(0,coord_transform(ymax,ρ,θ)))
    lanes.append([[xmin,ymin,xmax,ymax]])

    #print(lanes)

    # Draw lines on image
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img,lanes)
    return line_img


def hough_linesP(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Returns (x1,y1), (x2,y2) quadruplets
    """
    linesp = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                             minLineLength=min_line_len,
                             maxLineGap=max_line_gap)

    # Draw lines on image
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img,linesp,consolidate=True)

    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



########################################################################
# MAIN PROCESS FILE
########################################################################
def process_image(img0, plot=False):
    ny, nx, nz = img0.shape

    # Step 1: Do Gaussian blur on grayscale
    kernel_size = 3
    img1 = gaussian_blur(grayscale(img0), kernel_size)

    # Step 2: Get Canny Transform
    low = 50
    high = 100
    img2 = canny(img1,low,high)

    # Step 3: Mask out peripheral content
    a = [0.49, 0.58, 0.05, 0.95]                 # Factors [Htop, Vtop, HBottom, VBottom]
    vertices =[[[    a[0]*nx, a[1]*ny],    # Top left
                [    a[2]*nx, a[3]*ny],    # Bottom left
                [(1-a[2])*nx, a[3]*ny],    # Bottom right
                [(1-a[0])*nx, a[1]*ny]]]   # Top right
    img3 = region_of_interest(img2, np.array(vertices,dtype='int32'))

    # Step 4: Hough transform to detect lanes
    rho = 1               # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180   # angular resolution in radians of the Hough grid
    threshold = 10        # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10      # minimum number of pixels making up a line
    max_line_gap = 2      # maximum gap in pixels between connectable line segments
    img4 = hough_linesP(img3, rho, theta, threshold, min_line_len, max_line_gap)
    #img4 = hough_lines(img3, rho, theta, threshold)

    # Weighter average of original and lines
    img5 = weighted_img(img4, img0, α=0.8, β=5., λ=0.)

    # Draw on screen
    plt.ion()
    if plot:
        plt.imshow(img5)
        plt.draw()
        plt.pause(0.005)
        plt.show()

    return img5

def process_video(vidfile):
    white_output = vidfile.split('.')[0] + '_out.mp4'
    clip1 = VideoFileClip(vidfile)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    return

########################################################################
# MAIN SCRIPT FILE
########################################################################
if __name__ == '__main__':
    all_files = os.listdir("test_images/")
    for file in all_files:
        test_file = "test_images/" + file
        img_pre   = mpimg.imread(test_file)
        img_post  = process_image(img_pre,plot=True)
        time.sleep(1)

    process_video("solidWhiteRight.mp4")
    process_video("solidYellowLeft.mp4")
    process_video("challenge.mp4")
