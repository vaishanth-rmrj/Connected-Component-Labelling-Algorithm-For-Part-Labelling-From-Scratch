import cv2
import numpy as np
import matplotlib.pyplot as plt

class ConnectedComponentLabeler:
    def __init__(self, image):
        self.image = image
        self.image_copy = self.image.copy()
        self.img_width = self.image.shape[0]
        self.img_height = self.image.shape[1]

        self.label_img = np.ones(self.image.shape)
        self.label_count = 0

        self.equavalency_list = list()
        self.id = 0

    def get_neighbour_pixel(self, i, j):
        
        left_pixel = self.label_img[i-1,j] # left        
        top_pixel = self.label_img[i,j-1] # above
        return [left_pixel, top_pixel]

    def perform_labeling(self):

        # first scan
        for row in range(self.img_width):
            for col in range(self.img_height):

                if self.image[row, col] == [0]:
                    self.label_img[row, col] = 0
                
                else: 

                    # get the neighbour pixels
                    neighbour_pixels = self.get_neighbour_pixel(row, col)                    
                    if neighbour_pixels == [0, 0]:
                        self.label_count += 1
                        self.label_img[row, col] = self.label_count                

                    else:

                        if neighbour_pixels[0] == neighbour_pixels[1] or np.min(neighbour_pixels) == 0:
                            self.label_img[row, col] = np.max(neighbour_pixels)

                        else:
                            self.label_img[row, col] = np.min(neighbour_pixels)

                            if self.id == 0:
                                self.equavalency_list.append(neighbour_pixels)
                                self.id += 1
                            else:
                                check_count = 0
                                for k in range(self.id) :
                                    tmp = set(self.equavalency_list[k]).intersection(set(neighbour_pixels))
                                    if len(tmp) != 0 :
                                        self.equavalency_list[k] = set(self.equavalency_list[k]).union(neighbour_pixels)
                                        check_count += 1

                                if check_count == 0:
                                    self.id += 1
                                    self.equavalency_list.append(set(neighbour_pixels))      

                    cv2.circle(self.image_copy, (col, row), 1, (255), 1)
                cv2.imshow("Image", self.image_copy)
            if cv2.waitKey(10) == ord('q'):
                break

        # second scan
        for row in range(self.img_width):
            for column in range(self.img_height):
                for x in range(self.id):
                    if (self.label_img[row, column] in self.equavalency_list[x]) and self.label_img[row, column] !=0 :
                        self.label_img[row, column] = min(self.equavalency_list[x])     

        for row in range(self.img_width):
            for column in range(self.img_height):
                for x in range(self.id):
                    if (self.label_img[row, column] == min(self.equavalency_list[x])):
                        self.label_img[row, column] = x+1
                
        cv2.destroyAllWindows()

        return self.label_img, self.id

if __name__ == "__main__":
    coin_image = cv2.imread("images/parts.png", 0) # if error loading image, set path to: images/Q1image.png

    # thresholding the image
    _, thresh = cv2.threshold(coin_image, 127, 1, cv2.THRESH_BINARY)

    # performing morphological operations
    circular_kernel = np.array([[0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0]], dtype=np.uint8)

    erode = cv2.erode(thresh, circular_kernel, iterations=5)
    

    img = erode.copy()

    labeler = ConnectedComponentLabeler(img)
    label_img, label_count = labeler.perform_labeling()
    print("Number of labels", label_count)

    plt.figure(figsize=(50, 50))
    plt.subplot(2, 2, 1), plt.imshow(coin_image, cmap="gray")
    plt.title("Original Image")
    plt.subplot(2, 2, 2), plt.imshow(img, cmap="gray")
    plt.title("Eroded Image")
    plt.subplot(2, 2, 3), plt.imshow(label_img)
    plt.title("Labeled Image; Number of labels " + str(label_count))
    plt.show()

