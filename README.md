<img src="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/SCREEN-SHOTS/Heart-Disease-Prediction-System-banner.png" />

## Abstract 
<p> 
 Heart disease is currently one of the most common and serious health conditions worldwide. Unfortunately, its treatment can be expensive and often unaffordable for the average person. However, this issue can be partially addressed by detecting heart disease at an early stage through a Heart Disease Prediction System powered by Machine Learning (ML) and Data Mining techniques.Early prediction of heart-related issues can significantly improve the chances of effective treatment and recovery. In the healthcare and biomedical domain, vast amounts of data—such as text records, medical images, and test results—are generated, but much of it remains underutilized. By implementing a Heart Disease Prediction System, we can analyze this data effectively to support early diagnosis and reduce treatment costs while improving care quality.This system leverages patient data such as age, sex, blood pressure, cholesterol levels, and blood sugar levels to assess the likelihood of developing heart disease. It can identify complex patterns in patient data and assist in making intelligent medical decisions. To evaluate the system’s performance, a confusion matrix is used, which enables the calculation of key metrics such as accuracy, precision, and recall.

Overall, this Heart Disease Prediction System aims to deliver high performance, improved diagnostic accuracy, and cost-effective healthcare support. 
</p>

## Introduction
<p>
  The healthcare industry generates vast amounts of data, much of which contains hidden patterns that can be valuable for making informed and effective decisions. To extract meaningful insights and provide accurate predictions, advanced data mining techniques are essential.In this study, a Heart Disease Prediction System (HDPS) has been developed using Naive Bayes and Decision Tree algorithms to assess the risk level of heart disease in individuals. The system analyzes 13 medical parameters, including age, sex, blood pressure, cholesterol, and obesity, to predict the likelihood of a patient developing heart disease.
The HDPS not only offers risk predictions but also helps uncover significant relationships between medical factors and patterns associated with heart disease. To further enhance prediction accuracy, a Multilayer Perceptron (MLP) neural network with backpropagation is employed as a training algorithm.

The experimental results demonstrate that the proposed diagnostic system can effectively predict heart disease risk levels, supporting early intervention and informed medical decision-making.
</p>

### Aim
<p> 
  To predict heart disease according to input parameter values provided by user and dataset
stored in database.
</p>

### Objective
<p>
  The main objective of this project is to develop a heart disease prediction system. The system
can discover and extract hidden knowledge associated with diseases from a historical heart data
set Heart disease prediction system aims to exploit data mining techniques on medical data set
to assist in the prediction of the heart diseases.
</p>

### Project Scope
<p>
  The project has a wide scope, as it is not intended to a particular organization. This project is
going to develop generic software, which can be applied by any businesses organization.
Moreover it provides facility to its users. Also the software is going to provide a huge amount
of summary data.
</p>

## System Analysis
### Modules:
- **Patient Login:-** *Patient Login to the system using his ID and Password.*
- **Patient Registration:_** *If Patient is a new user he will enter his personal details and he
will user Id and password through which he can login to the system.*
- **My Details:-** *Patient can view his personal details.*
- **Disease Prediction:-** *- Patient will specify the input parameter values. System will take
input values and predict the disease based on the input data values specified by the
patient and system will also suggest doctors based on the locality*
- **Search Doctor:-** *Patient can search for doctor by specifying name, address or type.*
- **Feedback:-** *Patient will give feedback this will be reported to the admin*
- **Doctor Login:-** *Doctor will access the system using his User ID and Password.*
- **Patient Details:-** *Doctor can view patient’s personal details.*
- **Notification:-** *Admin and doctor will get notification how many people had accessed
the system and what all are the diseases predicted by the system.*
- **Admin Login:-** *Admin can login to the system using his ID and Password.*
- **Add Doctor:-** *Admin can add new doctor details into the database.*
- **Add Dataset:-** *Admin can add dataset file in database.*
- **View Doctor:-** *Admin can view various Doctors along with their personal details.*
- **View Disease:-** *Admin can view various diseases details stored in database.*
- **View Patient:-** *Admin can view various patient details that had accessed the system.*
- **View Feedback:-** *Admin can view feedback provided by various users.*
  
### Technology Used:
- #### Languages:
 
1) HTML
2) CSS
3) JAVASCRIPT
4) PYTHON

   
- #### FrameWork:
  - [BOOTSTRAP]
  - [DJANGO])
- #### Machine-Learning Algorithms:
  - <a href="https://en.wikipedia.org/wiki/Gradient_boosting">**GRADIENT BOOSTING ALGORITHM**</a>
  - <a href="https://en.wikipedia.org/wiki/Logistic_regression">**LOGISTIC REGRESSION**</a>
- #### ML/DL:
  - [NumPy]
  - [Pandas]
  - [scikit-learn]
- Database:
  - [PostreeSQL]
- #### IDE:
  - [VS Code]
  - [pyCharm]
- #### OS used for testing:
  - [MacOS]
  - [Ubuntu]
  - [Windows

## Run Locally

Clone the project

```bash
  git clone https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System
```

Go to the project directory

```bash
  cd Heart-Disease-Prediction-System
```

Start the server

```bash
  python manage.py runserver
```

## Model Training(Machine Learning)

```javascript
def prdict_heart_disease(list_data):
    csv_file = Admin_Helath_CSV.objects.get(id=1)
    df = pd.read_csv(csv_file.csv_file)

    X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
    nn_model = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1, random_state=0)
    nn_model.fit(X_train, y_train)
    pred = nn_model.predict([list_data])
    print("Neural Network Accuracy: {:.2f}%".format(nn_model.score(X_test, y_test) * 100))
    print("Prdicted Value is : ", format(pred))
    dataframe = str(df.head())
    return (nn_model.score(X_test, y_test) * 100),(pred)
```

### For a detailed Report <a href="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/REPORT/PYTHON%20CAPSTONE%20PROJECT%20REPORT%20(TEAM%202).pdf">Click Here</a>




## Output Screen-shots
When the application is runned then, a Welcome Page pops-up
<img src="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/SCREEN-SHOTS/WelcomePage.png" />

Admin Dash-board:
<img src="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/SCREEN-SHOTS/AdminDashboard.png" />

Entering Heart Details to check our Health:
<img src="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/SCREEN-SHOTS/AddHeartDetail.png" />

Since these details are stored in the Data-base, so we can also retrieve past results:
<img src="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/SCREEN-SHOTS/SearchLogs1.png" />

To view our own details:
<img src="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/SCREEN-SHOTS/ViewMyDetaile.png" />

If a user doesn't understand how to use the application then he can:
<img src="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/SCREEN-SHOTS/IntroductionViewVideo.png" />

To view registered Doctor information:
<img src="https://github.com/Kumar-laxmi/Heart-Disease-Prediction-System/blob/main/SCREEN-SHOTS/DoctorRecords.png" />


