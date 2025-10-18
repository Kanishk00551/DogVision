<img width="488" height="813" alt="image" src="https://github.com/user-attachments/assets/14b2f77b-9a58-4b2d-9861-ff5b41e5b792" />
<img width="439" height="182" alt="image" src="https://github.com/user-attachments/assets/3e2006ed-b0d4-4fd6-ae51-af98b6597476" />


# 🐶 DogVision - Dog Breed Classification using CNN

**DogVision** is a deep learning-based project that classifies images of dogs into their respective breeds using Convolutional Neural Networks (CNNs). The project aims to explore the capabilities of image classification models in the domain of computer vision and animal recognition.

---

## 📌 Project Overview

DogVision uses a custom CNN as well as transfer learning with pre-trained models (like **EfficientNetB0**) to identify dog breeds from image datasets. It includes data preprocessing, augmentation, model training, evaluation, and visualization steps in a clean and reproducible pipeline.

---

## 🧠 Techniques & Tools Used

* **Python**
* **TensorFlow / Keras** – for building and training CNNs
* **OpenCV** – for image processing
* **Matplotlib / Seaborn** – for data visualization
* **EfficientNetB0** – for transfer learning
* **Data Augmentation** – to improve model generalization

---

## 🧪 Key Features

* Custom-built CNN and fine-tuned pre-trained models
* Train-validation-test split with reproducible results
* Image preprocessing and augmentation pipeline
* Performance tracking using accuracy, confusion matrix, and learning curves

---

## 📈 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Custom CNN          | \~85%    |
| EfficientNetB0 (TL) | \~90%+   |

*Exact performance may vary depending on dataset splits and hardware.*

---

## ▶️ How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/Kanishk00551/DogVision.git
   cd DogVision
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add dataset**

   * Place your dataset inside the `data/` directory in the following format:

     ```
     data/
     ├── train/
     │   ├── breed1/
     │   ├── breed2/
     └── test/
     ```

4. **Train the model**

   ```bash
   python train.py
   ```

---

## 📌 Dataset

This project supports datasets with folder-based class separation (e.g., `ImageFolder` format). You can use public datasets like:

* [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* Or any Kaggle dog breed dataset

---

## 📬 Contact

For questions, suggestions, or collaborations:
**[Kanishk00551](https://github.com/Kanishk00551)**

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

