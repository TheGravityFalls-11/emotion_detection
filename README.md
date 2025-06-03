# 🎭 Emotion Detection using Deep Learning

This project detects human emotions from facial expressions using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset. It provides a user-friendly interface to upload images and predict the emotion displayed.

---

## 📌 Features
- 🔍 Real-time emotion detection from images
- 🧠 CNN-based deep learning model
- 📁 Trained on FER-2013 dataset
- 🌐 Simple web interface built using Flask

---

## 📂 Folder Structure
```
emotion_detection/
│
├── static/
│   └── style.css               # CSS styles
├── templates/
│   └── index.html              # HTML template
├── model/
│   └── model.pth               # Trained CNN model
├── app.py                      # Flask backend
├── train.py                    # Training script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🌍 Frontend Preview

<!-- Replace this with actual image link once uploaded -->
📷 **Frontend Screenshot:**
![Image](https://github.com/user-attachments/assets/7f3159e6-92de-453f-a3dd-526659207318)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/TheGravityFalls-11/emotion_detection.git
cd emotion_detection
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate     # For Windows
source venv/bin/activate  # For Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask App
```bash
python app.py
```

Go to `http://127.0.0.1:5000` in your browser to use the app.

---

## 🧠 Model Training

To retrain the model:

```bash
python train.py
```

Make sure the FER-2013 dataset is placed properly and the script is adjusted accordingly.

---

## 📦 Requirements

- Python 3.8+
- torch
- torchvision
- Flask
- PIL
- matplotlib
- numpy

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributions

Feel free to fork this repo and open a pull request to contribute.

---

## 📧 Contact

Made with ❤️ by [TheGravityFalls-11](https://github.com/TheGravityFalls-11)

