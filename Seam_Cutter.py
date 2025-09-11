import numpy as np     # we need this for array math and fast numerics
import matplotlib.pyplot as plt   #we need this for plotting the different views and seams from the image so we can track 
import imageio.v2 as imageio   #we need this for reading in the image and writing out modified images 


#First we need to calculate the energy map of the input image.
#To do this, we need to calculate the gradient of the image in both x and y directions.
#We can do this using the Sobel operator.
#The Sobel operator is a 3x3 matrix that is applied to the image to calculate the gradient in both x and y directions.
#The gradient is then the sum of the absolute values of the x and y gradients.
#magnitude sqrt(gx^2 + gy^2)

def energy_map(image): 
    assert image.ndim == 3 and image.shape[2] == 3  #setting up the dimensions of the 3 x 3 matrix
    imf = image.astype(np.float64)/255.0            #converting the image to a float and normalizing it
    #making it grey 
    grey = (0.299 * imf[...,0] + 0.587 * imf[...,1] + 0.114 * imf[...,2]) #converting the image to greyscale by averaging the RGB values
    gy, gx = np.gradient(grey)                      #calculating the gradient of the image in both x and y directions using np.gradient
    energy = np.sqrt (gx*gx + gy*gy)                # calculating the magnitude of the gradient using sqrt(gx^2 + gy^2)
    return energy.astype(np.float64)                #we return the energy map as a float so we have better precision


#Nowe we need to use some dynamic method to find the Vertical and Horizontal seams. 
#Each Verical we must add the Cells min of three parents from the previous row. 
#Each Horizontal we must add the Cells min of three parents from the previous column. or reuse vertical and transpose 

def cumulative_minimum_energy_map(energyImg, SD):
    assert energyImg.ndim == 2                       #setting up the dimensions of the 2 x 2 matrix
    dir_up = SD.upper() 

    if dir_up == 'VERTICAL':
        M, N = energyImg.shape
        Cumulative = np.zeros_like(energyImg, dtype=np.float64)  # "Dynamic programmuing to go row by row column by column"
        Cumulative[0] = energyImg[0]               # first row is the same as the energy image
        for i in range(1, M):                      ## then we fill in the rest row by row
            ##needs to be done for each parent pixel above
            left  = np.r_[np.inf, Cumulative[i-1, :-1]]   #ldefining the top left parents pixel / cell 
            up    = Cumulative[i-1]                       #for the diret top parent we simply look "up" 
            right = np.r_[Cumulative[i-1, 1:], np.inf]    # for the top right parent we look up and right 
            ##find the best parent 
            best = np.minimum(np.minimum(left, up), right) # find best parents between left and up then right aswell
            Cumulative[i] = energyImg[i] + best            # add the energy of the current pixel to the best parent
        return Cumulative                                  #return the cumulative energy map
    
    ## now we want direction checks to specify vertical or horizontal seams 
    if dir_up == 'HORIZONTAL':
        return cumulative_minimum_energy_map(energyImg.T, 'VERTICAL').T

    raise ValueError("Direction must be 'VERTICAL' or 'HORIZONTAL'")


##okay now we want to find the vertical seam first lets say 
def find_vert_seam(CumulativeEnergyMap):
    M, N = CumulativeEnergyMap.shape 
    #storing the pixels for each seam as a index 
    seam = np.zeros(M, dtype=np.int32)
    seam[-1] = int(np.argmin(CumulativeEnergyMap[-1]))     #find the last pixel of the seam by finding the min of the last row
    for i in range(M-2, -1, -1):                            #we start on the bottom and go up one by one 
        j = seam[i+1]                                       # finding the best next candidate 
        j_left  = max(j-1,0)
        j_right = min(j+1, N-1)
        ##all the candidates for the next pixel
        window = CumulativeEnergyMap[i, j_left:j_right+1]
        seam[i] = j_left + int(np.argmin(window))           #find the one with the lowest energy and choose it as the next pixel
    return seam                                             #return the new updated seam 


def find_horizontal_seam(cumulativeEnergyMap):              #we want to find the horizontal seam 
    return find_vert_seam(cumulativeEnergyMap.T).astype(np.int32)  #we reuse the logic we just laid out but tranpose with columns instead 


def decrease_width(image, energyImg_in):
    Cumulative = cumulative_minimum_energy_map(energyImg_in, 'VERTICAL')  #we want to find the vertical seam
    seam = find_vert_seam(Cumulative)                                     #find the best seams 
    M, N, _ = image.shape                                                 # find the size of the image but we dont care about the color channels
    out = np.zeros((M, N-1, 3), dtype=image.dtype)                        #create a new image with that column removed  
    for i in range(M):                                                    #for every row
        j = seam[i] 
        out[i,:,:] = np.delete(image[i, :, :], j, axis=0)                 #delete the column from the image interested only in i
    new_energy = energy_map(out)                                          #calculate the energy map of the new image
    return out, new_energy                                                ##see how much we reduced the energy map and the new image 
        

def decrease_height(image, energyImg_in):
    Cumulative = cumulative_minimum_energy_map(energyImg_in, 'HORIZONTAL')  #we want to find the vertical seam
    seam = find_horizontal_seam(Cumulative)                                  #find the best seams 
    M, N, _ = image.shape                                                   # find the size of the image but we dont care about the color channels
    out = np.zeros((M - 1 , N, 3), dtype=image.dtype)                       #create a new image with that column removed  
    for j in range(N):                                                      #for every column
        i = seam[j] 
        out[:,j,:] = np.delete(image[:, j, :], i, axis=0)                   #delete the row from the image only interested in j
    new_energy = energy_map(out)                                            #calculate the energy map of the new image
    return out, new_energy                                                  ##see how much we reduced the energy map and the new image 
