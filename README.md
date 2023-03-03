# Image Processing and height measurement API Overview
This project uses the Flask package to create an API that utilizes the YOLOv7 object detection algorithm to calculate water level from lines of detected parts of an image. The API returns parsed JSON containing confidence level, calculated water level, and bounding box dimensions.

## Requirements
To use this project, you will need to have the following installed:

- Python 3.x
- Flask package
- YOLOv7 object detection algorithm
- OpenCV

## Installation

  * Clone the repository to your local machine.
  * Install the required packages using pip install -r requirements.txt.
  * Download the pre-trained YOLOv7 model.
  * Run the API using python app.py.
  
## Usage
  * Send an image to the API using the /ImageProcess endpoint.
  * The API will return a JSON response containing confidence level, calculated water level, and bounding box dimensions.
