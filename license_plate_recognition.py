import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to detect license plate
def detect_plate(img):
    plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')  # Adjust the path as needed
    plate_img = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)

    if len(plate_rect) > 0:
        for (x, y, w, h) in plate_rect:
            roi = plate_img[y:y+h, x:x+w]
            cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        return plate_img, roi
    else:
        return plate_img, None

# Function to find contours
def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or 15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of the rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(intX)  # stores the x coordinate of the character's contour, to be used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with a black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # Return characters in ascending order with respect to the x-coordinate (most-left character first)

    plt.show()
    # arbitrary function that stores a sorted list of character indices
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

# Function to segment characters
def segment_characters(img):
    # Extract license plate region
    output_img, plate = detect_plate(img)

    if plate is not None:
        # Preprocess cropped license plate image
        img_lp = cv2.resize(plate, (333, 75))
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]

        # Make borders white
        img_binary_lp[0:3, :] = 255
        img_binary_lp[:, 0:3] = 255
        img_binary_lp[72:75, :] = 255
        img_binary_lp[:, 330:333] = 255

        # Estimations of character contours sizes of cropped license plates
        dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]
        plt.imshow(img_binary_lp, cmap='gray')
        plt.show()
        cv2.imwrite('contour.jpg', img_binary_lp)

        # Get contours within cropped license plate
        char_list = find_contours(dimensions, img_binary_lp)

        return char_list
    else:
        return None


def predict_characters(char_images, model):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for ch in char_images:
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.zeros((28, 28, 3))
        for i in range(3):
            img[:, :, i] = img_
        img = img.reshape(1, 28, 28, 3)
        probabilities = model.predict(img)[0]
        predicted_class = np.argmax(probabilities)
        character = dic[predicted_class]
        output.append(character)

    return output
    pass

# Predict characters using the loaded model
#predicted_characters = predict_characters(char, loaded_model)

if __name__ == "__main__":
    image_path = input("Enter the path of the license plate image: ")
    char = segment_characters(image_path)
    print("Segmented characters:", len(char))
