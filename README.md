# 🦴 Bone Fracture Detection using YOLOv11 

## 📌 Project Overview

This project implements a Bone Fracture Detection System using YOLOv11 (Ultralytics). The system detects and localizes fracture regions in X-ray images using bounding box prediction with confidence scores.

Unlike traditional CNN-based classification models, this system performs object detection to precisely highlight fracture areas instead of just predicting "Fractured" or "Normal".

---

## 🚀 Key Features

* Bounding box localization of fracture regions
* Confidence score display
* Flask-based web interface
* Secure image upload system
* Automatic image preprocessing
* Clean UI for visualization

---

## 🧠 Model Details

* Model: YOLOv11 (Ultralytics)
* Framework: PyTorch (Backend)
* Detection Type: Object Detection
* Output: Bounding Box + Confidence Score

---

## 🏗️ Project Structure

```
YOLOv11/
│
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── uploads/
│   └── results/
├── model/
│   └── best.pt 
└── .gitignore
```

---

## 🛠️ Technologies Used

* Python
* YOLOv11 (Ultralytics)
* PyTorch
* OpenCV
* Flask
* HTML / CSS

---

## 📊 How It Works

1. User uploads an X-ray image.
2. Image is preprocessed (resizing & normalization).
3. YOLOv11 model performs fracture detection.
4. Bounding boxes are drawn on detected fracture regions.
5. Processed image is displayed with confidence score.

---

## 👩‍💻 Developed By

-**Kaya Dhankar**
-B.Tech CSE (Artificial Intelligence)

---

## 📷 Sample Output
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/be5943db-de4a-44e1-b776-51276f5d9e71" />

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/8deef441-2db0-4eca-9bf3-6657eb619579" />

