# BLRTraffic

This Repository contains the code for Autamatic Number Plate License Number Detection. This happens in two stages
  1. License Plate Detection
     * YOLOv8 is trained on detecting the number plates
  2. Number on the Plate Recognition
     * Run PaddleOCR or EasyOCR to extract numbers/text from the detected license plates

Steps to Use this Repository
1. Clone this repo
   ```git clone https://github.com/campusguru/BLRTraffic```
3. create new conda environment
   ```conda create --name BLRTraffic python=3.9```
4. Install Dependencies
    ```pip3 install -r requirements.txt```
5. Run the file for inference
    ```python app.py```
   or
   ```python inference.py```

