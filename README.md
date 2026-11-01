# AI-Powered Kidney Stone Detection

## 📋 Project Overview
This project uses deep learning to detect kidney stones from ultrasound images. It achieves high accuracy using transfer learning with VGG16 architecture.

## 🎯 Features
- Upload ultrasound images for analysis
- Real-time detection with confidence scores
- Patient data management with Firebase
- Modern, responsive web interface
- Detailed analysis reports with stone localization
- Heatmap visualization of stone locations
- Bounding box detection
- Comprehensive text reports

## 🛠️ Technologies Used
- **Python 3.10** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **VGG16** - Pre-trained CNN architecture
- **Flask** - Web framework
- **Firebase** - Authentication & Realtime Database
    - *Make sure a Realtime Database instance is created in your Firebase console.*
    - Update `app.py` (or `firebase_config.json`) with the correct `databaseURL`; a wrong URL leads to 404 errors during registration.
- **HTML5/CSS3/JavaScript** - Frontend development
- **Google Colab** - Model training
- **OpenCV** - Image processing and visualization

## 📊 Model Performance
- **Training Images**: 9,440 (4,422 Normal, 5,018 Stone)
- **Accuracy**: 100% on validation set
- **Processing Time**: <2 seconds per image
- **Loss**: Near-zero (3e-07)

## 🚀 How to Run

### Local Deployment
1. Install dependencies:
pip install -r requirements.txt

text

2. Run the Flask app:
python app.py

text

3. Open browser and navigate to `http://localhost:5000`


## 📁 Project Structure
```
Kidney_Stone_Detection_Project/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── model/                 # Trained model files (best_model.keras)
├── templates/             # HTML templates for pages
├── static/                # CSS & JavaScript assets
├── uploads/               # User-uploaded images (ignored by Git)
├── firebase_config.json   # Firebase project settings
└── serviceAccountKey.json # Firebase credentials (private)
```

## 👨‍💻 Created By
**Danish Khan**
- Final Year Computer Science Student
- Project for College Submission

## 📝 License
This project is created for educational purposes as part of a college project.
## ⚠️ Note
This is a college project and should not be used for actual medical diagnosis without proper validation and certification.

## 📞 Contact
For any queries about this project, please contact:
- [Email](danish1442003@gmail.com)
- [instagram](https://www.instagram.com/danish.khann_14/?hl=en)

## 🙏 Acknowledgments
- VGG16 pre-trained model from Keras
- Mendeley Data for the ultrasound dataset
- Firebase for database services