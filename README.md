# SCT_ML_3
SVM model for classifying Cats and Dogs using image dataset
# 🐶 Cats vs Dogs Image Classifier using SVM

This project implements an image classifier to distinguish between cats and dogs using a **Support Vector Machine (SVM)** algorithm. The model is trained on the popular **Dogs vs Cats** dataset from Kaggle and leverages basic image processing and classical machine learning techniques.

---

## 📌 Project Overview

- 🔍 **Objective**: Classify images as either **cat** or **dog**
- ⚙️ **Algorithm**: Support Vector Machine (SVM) with a linear kernel
- 🖼️ **Image Preprocessing**: Resized all images to 64x64 pixels and normalized pixel values
- 📊 **Evaluation**: Accuracy, confusion matrix, classification report, and sample visualizations

---

## 🧠 Technologies & Libraries Used

| Tool / Library      | Purpose                              |
|---------------------|--------------------------------------|
| Python              | Programming Language                 |
| OpenCV              | Image loading and resizing           |
| NumPy               | Numerical operations                 |
| Matplotlib, Seaborn | Visualizations and plots             |
| Scikit-learn        | Machine learning model and metrics   |

---

## 📁 Dataset

We used the [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) provided by Kaggle.

Due to GitHub's file size limitations, the dataset is **not included** in this repository. Please download and extract the dataset manually.

> 💡 **Expected directory structure** after extraction:

cats-vs-dogs-svm/
├── svm_cats_dogs.py
├── requirements.txt
├── README.md
└── dataset/
└── training/
├── cats/
│ ├── cat.1.jpg
│ ├── cat.2.jpg
│ └── ...
└── dogs/
├── dog.1.jpg
├── dog.2.jpg
└── ...

yaml
Copy
Edit

---

## 🚀 Getting Started

Follow these steps to run the project locally:

### 1. Clone the Repository


git clone https://github.com/yourusername/cats-vs-dogs-svm.git
cd cats-vs-dogs-svm
2. Install Dependencies
Make sure you have Python 3.x installed. Then run:

bash
Copy
Edit
pip install -r requirements.txt
3. Add the Dataset
Download the dataset from Kaggle Dogs vs Cats

Extract the training images to dataset/training/cats and dataset/training/dogs as shown above.

4. Run the Model
bash
Copy
Edit
python svm_cats_dogs.py
📈 Output & Evaluation
The model produces the following:

✅ Overall accuracy score

🔢 Classification report with precision, recall, and F1-score

📊 Confusion matrix heatmap

📉 Bar graph showing model accuracy

🖼️ Image preview of 5 predictions with true and predicted labels

📋 Example Output
yaml
Copy
Edit
Training SVM...
✅ Accuracy: 89.50%

✅ Classification Report:
              precision    recall  f1-score   support
         Cat       0.88      0.87      0.87       100
         Dog       0.91      0.92      0.91       100
    accuracy                           0.90       200

pip install -r requirements.txt
🧩 Future Enhancements
Switch to Convolutional Neural Networks (CNNs) for improved accuracy

Integrate GUI for uploading and classifying custom images

Add training/test split from the original dataset

Deploy as a web app using Flask or Streamlit

🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

📬 Contact
Author: Nandini Nandu
LinkedIn: linkedin.com/in/nanditha-sn-60b278372

⭐ Show Your Support
If you found this project helpful, please consider starring the repository and sharing it with others!
