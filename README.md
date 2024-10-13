# Orange ID Detection Program

This is a simple **Orange ID Detection** program that identifies and detects oranges in an image using basic image processing techniques. The program uses **OpenCV** and **Numpy** to process the image and detect circular shapes representing oranges.

## Features

- Detects oranges in images by identifying circular shapes.
- Uses image processing techniques like grayscale conversion, Gaussian blur, and Hough Circle detection.
- Displays the detected oranges by drawing circles around them in the image.

## Technologies Used

- **Python**: The programming language used to build this program.
- **OpenCV**: For image processing and orange detection.
- **Numpy**: For numerical operations and array manipulations.

## How It Works

1. The program converts the input image to grayscale.
2. It applies a **Gaussian Blur** to reduce noise and smooth the image.
3. Using the **Hough Circle Transform** method, it detects circular shapes, which are identified as oranges.
4. The program draws circles around detected oranges and displays the result.

## Prerequisites

Make sure you have Python installed along with the following libraries:
- OpenCV: `pip install opencv-python`
- Numpy: `pip install numpy`

## How to Run

1. Clone the repository or download the script.
2. Install the required dependencies: `pip install -r requirements.txt`.
3. Place an image of oranges in the same directory.
4. Run the program using the command:
   ```bash
   python orange_id_detection.py
