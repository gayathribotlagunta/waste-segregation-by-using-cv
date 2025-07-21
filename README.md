# waste-segregation-by-using-cv
# 🗑️ Waste Segregation Using Computer Vision

This project leverages computer vision and machine learning to automate waste segregation. It classifies waste into categories like biodegradable, non-biodegradable, and recyclable using image classification techniques. The system aims to support sustainable waste management through AI-driven sorting.

## 🔍 Problem Statement

Manual waste sorting is time-consuming and error-prone, often leading to improper disposal and environmental damage. This project provides a smart, vision-based solution to identify and segregate waste items efficiently.

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – Image processing and contour detection
- **TensorFlow / Keras** – Model training and prediction
- **NumPy & Pandas** – Data manipulation
- **Matplotlib / Seaborn** – Data visualization (EDA)
- **Streamlit / Flask** *(optional)* – For a simple web interface (if used)

## 📁 Project Structure

waste-segregation-cv/
├── dataset/ # Images of different waste categories
├── model/ # Trained model weights
├── notebooks/ # Jupyter notebooks for EDA and training
├── src/ # Python scripts (training, preprocessing, prediction)
│ ├── preprocess.py
│ ├── train_model.py
│ └── predict.py
├── app.py # Streamlit or Flask app (if applicable)
├── requirements.txt # Dependencies
└── README.md # Project documentation

## 📊 Dataset

A labeled image dataset of various waste types:
- **Biodegradable:** food, paper, plant waste
- **Non-Biodegradable:** plastic, glass, metal
- **Recyclable:** cans, bottles, cardboard

> Dataset may be sourced from custom image collection or open datasets on [Kaggle](https://www.kaggle.com/), [Roboflow](https://roboflow.com), etc.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gayathribotlagunta/waste-segregation-cv.git
   cd waste-segregation-cv
pip install -r requirements.txt
streamlit run app.py
📈 Model & Accuracy
Architecture: CNN (Convolutional Neural Network)

Training Accuracy: ~94%

Validation Accuracy: ~91%

Loss and accuracy plots are available in the /notebooks folder

📸 Sample Output

🧠 Future Improvements
Real-time classification using a webcam

Integration with smart bins

Expansion to more waste types

Edge deployment on devices like Raspberry Pi

🤝 Contributing
Contributions are welcome! Fork this repo, create a new branch, and submit a pull request.

📄 License
This project is open-source under the MIT License.

📬 Contact
GitHub: gayathribotlagunta

Email: gayathribotlagunta28@gmail@example.com

