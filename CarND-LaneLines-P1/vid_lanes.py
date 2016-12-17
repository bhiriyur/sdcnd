import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
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


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
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
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold):
    """ Returns the hough lines"""
    
    # Returns (rho,theta) pairs, Using a multiscale hough-transform
    lines  = cv2.HoughLines( img, rho, theta, threshold, np.array([]),1,5).squeeze()

    # Discard Horizontal lines (or close to horizontal)
    discard = []
    for i in range(lines.shape[0]):        
        if abs(lines[i,1]-np.pi/2)<0.1:
            discard.append(i)

    np.delete(lines,discard)


def hough_linesP(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """   
    # Returns (x1,y1), (x2,y2) quadruplets
    linesp = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                             minLineLength=min_line_len, maxLineGap=max_line_gap).squeeze()

    # Discard Horizontal lines (or close to horizontal)
    discard = []
    for i in range(linesp.shape[0]):        
        if abs(linesp[i,0,1]-linesp[i,0,3])<20:
            discard.append(i)
    linesp = np.delete(linesp,discard,axis=0)
    
    # Draw lines on image
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, linesp)

    # Get min and max y extents
    x1min,y1min,x2min,y2min = np.squeeze(np.array(linesp)).min(axis=0)
    ymin = np.min((y1min,y2min))
   

    return line_img, lines.squeeze(), ymin

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

def consolidate_extend(img,lines,nbins,ymin):
    """ Consolidate to nbins, extend to ymin and add lines to 
    original img.
    """
   
    # Setup bins
    rmin, tmin = lines.min(axis=0)
    rmax, tmax = lines.max(axis=0)
    bin_edges = np.linspace(tmin,tmax,nbins+1)
    rbins = [[] for i in range(nbins)]
    tbins = [[] for i in range(nbins)]

    #print('bin_edges = {}'.format(bin_edges))
   
    # Consolidate to bins
    for i in range(lines.shape[0]):        
        for j in range(nbins):
            # Skip extremely flat lines
            #if lines[i,1]>0.8*np.pi/2:
            #    continue
            
            if lines[i,1] >= bin_edges[j] and lines[i,1] < bin_edges[j+1]:
                rbins[j].append(lines[i,0])
                tbins[j].append(lines[i,1])

    # Average for each bin
    for i in range(nbins):
        #print('i = {}, rho = {}, theta = {}'.format(i,rbins[i],tbins[i]))
        rbins[i] = np.mean(rbins[i])
        tbins[i] = np.mean(tbins[i])

    # Convert rho,theta to (x1,y1,x2,y2)
    new_lines = []
    ymax = img.shape[0]
    for rho, theta in zip(rbins,tbins):
        xmin = int((rho - ymin*np.sin(theta))/np.cos(theta))
        xmax = int((rho - ymax*np.sin(theta))/np.cos(theta))
        new_lines.append([[xmin,ymin,xmax,ymax]])

    # Draw lines on image
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, new_lines)

    return line_img
    

########################################################################
# MAIN PROCESS FILE
########################################################################
def process_image(img0, plot=True):
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
    rho = 1             # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180   # angular resolution in radians of the Hough grid
    threshold = 20       # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10   # minimum number of pixels making up a line
    max_line_gap = 1   # maximum gap in pixels between connectable line segments
    img4, lines, ymin = hough_lines(img3, rho, theta, threshold, min_line_len, max_line_gap)
    
    # Step 5: Consolidate into nbins and extend lines
    nbins = 2  
    img5 = consolidate_extend(img4,lines,nbins,ymin)

    # Weighter average of original and lines
    img6 = weighted_img(img4, img0, α=0.8, β=5., λ=0.)

    # Draw on screen
    plt.ion()
    if plot:
        plt.imshow(img6)        
        plt.draw()
        plt.pause(0.05)
        plt.show()

    return img6

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
    test_file = "test_images/" + all_files[0]
    img_pre   = mpimg.imread(test_file)
    img_post  = process_image(img_pre)

    process_video("solidWhiteRight.mp4")
    process_video("solidYellowLeft.mp4")
    process_video("challenge.mp4")
