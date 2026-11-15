# 🚗 Shipping Cost Prediction

## Problem statement
In this data science project, you will build a machine learning system which will be able predict the cost of the shipment or package by using machine learning algorithms. This project will be very usefull for logistics companies, where on a day to day basis a lot of couriers,packages or goods are transported via different modes of transport. This main concern with these logistics companies, is trying to deliver these goods in an efficient and cost efficient way as possible, so pricing of the shipment is tricky and involves a lot of variables to consider while pricing of the shipment. There might be scenarios where the shipment might be delayed due to some external reasons, leading to loss for the company and delay in delivery of the shipment. So logistics companies need to use dynamic pricing based on several factors and variables to price the shipment in such a way that there are no loss to the company and price of the shipment is as less as possible so that customers can use thier services more due to effective pricing rates.

## Solution Proposed
Now the question is how to dynamically predict prices of the particular shipment ?. One of the approaches which we can use of machine learning approach, where we can predict the shipping price based on the domain knowledge and leverage previous shipment data to predict the prices.

The problem is to reduce the cost due to unnecessary repairs. So it is required to minimize the false predictions.

## Dataset used
[Dataset Link][https://www.kaggle.com/code/klmsathishkumar/shipping-cost-prediction/data]

## Tech Stack Used
1. Python 
2. FastAPI 
3. Machine learning algorithms
4. Docker
5. MongoDB

## Infrastructure required
1. AWS S3
2. Azure
3. Github Actions

## How to run
Before you run this project make sure you have MongoDB Atlas account and you have the shipping dataset into it.

Step 1. Cloning the repository.
```
git clone https://github.com/Machine-Learning-01/shipment-pricing-prediction.git
```
Step 2. Create a conda environment.
```
conda create --prefix venv python=3.7 -y
```
```
conda activate venv/
````
Step 3. Install the requirements 
```
pip install -r requirements.txt
```
Step 4. Export the environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

export MONGODB_URL="mongodb+srv://<username>:<password>@ineuron-ai-projects.7eh1w4s.mongodb.net/?retryWrites=true&w=majority"

```
Step 5. Run the application server
```
python app.py
```
Step 6. Train application
```bash
http://localhost:8080/train
```
Step 7. Prediction application
```bash
http://localhost:8080/predict
```
## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image
```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> --build-arg MONGODB_URL=<MONGODB_URL> . 
```

3. Run the Docker image
```
docker run -d -p 8080:8080 <IMAGE_NAME>
```
## Project Architecture - 
![WhatsApp Image 2022-09-22 at 15 29 19](https://user-images.githubusercontent.com/71321529/192722336-54016f79-89ef-4c8c-9d71-a6e91ebab03f.jpeg)

## Data Collection Architecture -
![WhatsApp Image 2022-09-22 at 15 29 10](https://user-images.githubusercontent.com/71321529/192721926-de265f9b-f301-4943-ac7d-948bff7be9a0.jpeg)

## Deployment Architecture -
![WhatsApp Image 2022-09-22 at 15 29 04](https://user-images.githubusercontent.com/71321529/192722300-b906b222-63f7-452b-8e30-e234405031f2.jpeg)

## Models Used
* Linear Regression
* Lasso Regression
* Ridge Regression
* K-Neighbors Regressor
* Decision Tree
* Random Forest Regressor
* XGBRegressor
* CatBoosting Regressor
* AdaBoost Regressor

From these above models after hyperparameter optimization we selected Top two models which were XGBRegressor and Random Forest Regressors and used the following in Pipeline.

* GridSearchCV is used for Hyperparameter Optimization in the pipeline.

## `shipment` is the main package folder which contains 


**Components** : Contains all components of Machine Learning Project
- DataIngestion
- DataValidation
- DataTransformations
- ModelTrainer

**Custom Logger and Exceptions** are used in the Project for better debugging purposes.

### For infra setup refer this docs folder in the repository

## Conclusion
- This Project can be used in real-life by Users.