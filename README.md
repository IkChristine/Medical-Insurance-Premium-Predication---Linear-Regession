# Medical-Insurance-Premium-Regression Analysis Using Artificial Neural Networks (ANN)

This case study aims to build, train and test a machinie learning model to predict insurance cost based on customer features such as age, gender, Body Mass Index (BMI), number of children, smoking habits, and geo-location.

#### Objectives:
1. Perform data cleaning, feature engineering and visualization
2. Build, train and test an artificial neural network model in Keras and Tensorflow
3. Understand the theory and intuition behind artificial neural networks


#### Available features include:
* Inputs: 
1. age: Customer's age
2. sex : Insurance contractor gender
3. bmi: Body Mass Index (18.5 to 24.9 for ideal bmi)
4. Children: Number of children covered by health insurance/number of dependents
5. smoker: Smoking habit of customers
6. region: The beneficiary's residential area in the US, Northeast, Southeast, Southwest, Northwest
* Target (output):  
1. charges: Individual medical costs billed by health insurance



#### Data Source: https://www.kaggle.com/datasets/mirichoi0218/insurance

<p>&nbsp;</p>

## TRAINING AND EVALUATING A LINEAR REGRESSION MODEL IN SCIKIT-LEARN
Linear Regression Model achileved a 69% accuracy score.
- RMSE = 6536.847 
- MSE = 42730370.0 
- MAE = 4555.098 
- R2 = 0.6953286415758744 
- Adjusted R2 = 0.6859179432461717

- The coefficient of determinatio (R2) was 69%. This means that 69% of the variations in the output (Charges) was represented by the variations in the input features.
- This is a reasonable score however we can still attempt to increase the score and get it closer to 100%.

<p>&nbsp;</p>

## TRAINING AND EVALUATING AN ARTIFICIAL NEURAL NETWORK (ANN) BASED REGRESSION MODEL (Keras API)
- ANN_model = keras.Sequential() 
- resulted in about 38,351 artificially trainable parameters to optimize
ANN_model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_hist = ANN_model.fit(X_train, y_train, epochs = 100, batch_size= 20, validation_split= 0.2)  - Accuracy : 0.7409380674362183
- Resulted to an acciracu score of about 74%

<p>&nbsp;</p>

![b885aaf9-831c-45cf-92fd-44f4368c16cb](https://github.com/IkChristine/Medical-Insurance-Premium-Predication-using-ML-Linear-Regression-/assets/104997783/c45c667f-f8a2-48e8-a19a-42cc34bc4559)

- The Validation error tends to increase slightligh, showing the model is overfiting the training data, however, model still performed quite well.
- In essence, performance on training data was good, but not great on test data.

<p>&nbsp;</p>

![164cc399-4407-4068-8ecc-64b22af5dfbe](https://github.com/IkChristine/Medical-Insurance-Premium-Predication-using-ML-Linear-Regression-/assets/104997783/5d151e37-6d29-4083-963a-32442d1d73ec)

- Beyond 20,000, the model predictions did not accurately match the True values (test data set).

- RMSE = 6161.465 
- MSE = 37963650.0 
- MAE = 3815.8079 
- R2 = 0.7293158320104738 
- Adjusted R2 = 0.7209549310687123

- The coefficient of determinatio (R2) is 72%, which slighlty more accurate compared to the linear regresion model with scikit learn.

<p>&nbsp;</p>

## Dropout layers added to improve the network generalization ability
ANN_model.add(Dropout(0.5)
- After Dropout layers were added the Accuracy score was 80.4% and there was less overfit.

![8b42ebba-076c-40ff-b3d4-0ed1612cb32e](https://github.com/IkChristine/Medical-Insurance-Premium-Predication-using-ML-Linear-Regression-/assets/104997783/27ee3249-6032-4f0b-b7e2-8c505be5fe60)


![583ade7f-f718-4171-a350-d17799225355](https://github.com/IkChristine/Medical-Insurance-Premium-Predication-using-ML-Linear-Regression-/assets/104997783/6e7c78fe-2fc7-4c78-9e32-675b02c10ed3)


- RMSE = 5362.887 
- MSE = 28760556.0 
- MAE = 3271.4172 
- R2 = 0.7949346600648356 
- Adjusted R2 = 0.7886005955108537


**After dropout was introduced, the coefficient of determination (R2) became **79.5%** compared to 72.9% before dropout.**
