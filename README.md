# ⚽ BallTracker Project

## 📌 Introduction
This project aims to **detect and track football players, referees, and the ball** in match videos using **YOLO**, one of the leading object detection models. The system enhances detection accuracy through training and assigns players to teams based on **t-shirt color clustering via KMeans**.

Key features include:
- Team identification using **KMeans clustering** on pixel data.
- **Ball possession analytics** by tracking which team controls the ball.
- **Optical Flow** to measure **camera movement** across frames.
- **Perspective transformation** to calculate movement in **real-world units (meters)**.
- **Speed and distance tracking** for individual players.

This project showcases a combination of computer vision and machine learning techniques, making it valuable for both beginners and experienced engineers.

![Screenshot](output_videos/screenshot.png)

---

## 🧰 Modules Used
- **YOLO (v5)**: Object detection (players, referees, ball)
- **KMeans**: T-shirt color clustering (team assignment)
- **Optical Flow**: Camera motion estimation
- **Perspective Transformation**: Real-world distance estimation
- **Speed/Distance Calculation**: Player movement tracking

---

## 📦 Trained Models
Download the trained YOLOv5 model:

- [Trained YOLOv5 Model](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)  
  → Place it in the `models/` folder. 

---

## 🎥 Sample Input Video
Download the sample match video:

- [Sample Input Video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)  
  → Place it in the `input_videos/` folder.

---

## ✅ Requirements

To run this project, make sure you have the following installed:

- Python 3.x

### Install dependencies:
```bash
pip install -r requirements.txt
