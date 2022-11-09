## Midterm Project mlzoomcamp Energy Efficiency of Buildings

## Table of Contents
 * [Description of the problem](#description-of-the-problem)
 * [Project Objectives](#project-objectives)
 * [Local deployment](#local-deployment)
 * [Production deployment](#production-deployment)
   * [Docker container](#docker-container)
   * [Cloud deployment](#cloud-deployment)
 * [More](#what-else-can-i-do)


Repo contains the following:

* `README.md` with
  * Description of the problem
  * Instructions on how to run the project
* Data `ENB2012_data.csv`
* `notebook.ipynb` a Jupyter Notebook with the data analysis and models
* Script `train.py` (suggested name)
  * Training the final model
  * Saving model with BentoML
* Script `predict.py` (suggested name)
  * Loading the model
  * Serving it via a web serice
* Script `locustfile.py`
  * Test the service
* Json files `test.json` and `test2.json` to test the service. Change them to produce another prediction.
* Files with dependencies
  * `Pipenv` and `Pipenv.lock` if you use Pipenv
  * `bentofile.yaml` required for BentoML
* `Dockerfile` for running the service
* Instructions for Production deployment
  * Video or image of how you interact with the deployed service


## Description of the problem


AI techniques such as machine and deep learning are increasingly and successfully being used to build solutions for the built environment.

The energy consumption of heating, ventilation, and air-conditioning systems, particularly for cooling and heating loads, is a major problem of energy-efficient buildings in smart cities.


We will investigate the Energy Efficiency [Dataset](https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset)

based on paper

A. Tsanas, A. Xifara: 'Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools', Energy and Buildings, Vol. 49, pp. 560-567, 2012

On residential buildings, this study offers an Improved Harris Hawks Optimization with Hybrid Deep Learning Based Heating and Cooling Load Prediction (IHHOHDL-HCLP) model. 

This model can help energy-efficient buildings improve the performance of their air-conditioning systems and benefit the environment in the long run.


Abstract

We develop a statistical machine learning framework to study the effect of eight input variables (relative
compactness, surface area, wall area, roof area, overall height, orientation, glazing area, glazing area
distribution) on two output variables, namely heating load (HL) and cooling load (CL), of residential
buildings. We systematically investigate the association strength of each input variable with each of
the output variables using a variety of classical and non-parametric statistical analysis tools, in order to
identify the most strongly related input variables. Then, we compare a classical linear regression approach
against a powerful state of the art nonlinear non-parametric method, random forests, to estimate HL and
CL. Extensive simulations on 768 diverse residential buildings show that we can predict HL and CL with
low mean absolute error deviations from the ground truth which is established using Ecotect (0.51 and
1.42, respectively). The results of this study support the feasibility of using machine learning tools to
estimate building parameters as a convenient and accurate approach, as long as the requested query
bears resemblance to the data actually used to train the mathematical model in the first place.
    
    
The dataset contains eight attributes (or features, denoted by X1…X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.

Specifically:

* X1 Relative Compactness
* X2 Surface Area - m²
* X3 Wall Area - m²
* X4 Roof Area - m²
* X5 Overall Height - m
* X6 Orientation - 2:North, 3:East, 4:South, 5:West
* X7 Glazing Area - 0%, 10%, 25%, 40% (of floor area)
* X8 Glazing Area Distribution - 1:Uniform, 2:North, 3:East, 4:South, 5:West
* y1 Heating Load - kWh
* y2 Cooling Load - kWh


Paper Highlights

We study the effect of eight common building parameters on heating and cooling load. 

A robust statistical machine learning methodology is presented. 

Accurate predictions compared to ground truth: 0.51 for heating load, 1.42 for cooling load

## Project Objectives

This mid-project will provide a first-hand experience using ML.

On the given dataset, a regression model was trained to predict the target features 'Heating Load' and 'Cooling Load'. To begin, we performed exploratory data analysis (EDA) to better understand our dataset. For feature selection, we analyzed the association between our features and targets and obtained a deeper understanding of our targets.

Multiple models were trained:

* Linear Regression
* Ridge Regression
* Decision Tree
* Random Forest
* Xgboost

and optimized hyperparameters to produce a better model (based on the RMSE criterion, lower the better) from a base model. The best model (xgboost) was trained and saved with BentoML in order to be deployed.

## Local deployment

All development was done on Windows with conda.

```bash
conda activate ml-zoomcamp
```

1. Download repo
```bash
git clone https://github.com/dimzachar/mlzoomcamp_projects.git
```

2. For the virtual environment, I utilized pipenv. If you want to use the same venv as me, install pipenv and dependencies, navigate to the folder with the given files:
```bash
pip install pipenv
pipenv install numpy pandas seaborn bentoml scikit-learn==1.1.3 xgboost==1.7.1 pydantic==1.10.2
```

3. Enter shell and run train.py

```bash
pipenv shell
pipenv run python train.py
```

![cli](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/cli.png)

4. Then, get the service running on [localhost](http://localhost:3000)

```bash
pipenv run bentoml serve predict.py:svc
```

and test it with the data in the `test.json` and `test2.json`

![service](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/BentoML-Prediction-Service%20(4).png)
![service2](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/BentoML-Prediction-Service%20(1).png)
![service3](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/BentoML-Prediction-Service%20(2).png)
![service4](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/BentoML-Prediction-Service%20(3).png)

The output here is showing the total load and prints low if heating+cooling load is below the 25th percentile, average if between 25th and 75th percentile or high if it is greater than 75th percentile. You can check the exact prediction on the cli.


Optional: Run locust to test server, make sure you have installed it
```bash
pipenv run locust -H http://localhost:3000
```

and check it out on [browser](http://localhost:8089)

![locust](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Locust-for-locustfile-py.png)

## Production deployment with BentoML

You need to have Docker installed (I used Docker Desktop with WSL2) 
* [Docker Installation Windows](https://docs.docker.com/desktop/install/windows-install)

and open for the following:

First we need to build the bento with

```bash
pipenv run bentoml build
```

### Docker container

Once we have the bento tag we containerize it

```bash
pipenv run bentoml containerize  energy_efficiency_regressor:tag --platform=linux/amd64
```

Replace tag with the tag you get from bentoml build.

![cli3](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/cli3.png)

### Cloud deployment 

In order to deploy it to AWS we push the docker image. Make sure you have an account and install AWS CLI. Instructions can be found [here](https://mlbookcamp.com/article/aws) and the instructions to publish are described [here](https://www.youtube.com/watch?v=aF-TfJXQX-w&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=72).

First, reate a repo on Amazon Elastic Container Registry (ECR) with an appropriate name
![registry2](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Elastic-Container-Registry%20(2).png)

![registry](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Elastic-Container-Registry%20.png)

You will find the push commands there to push the docker image
![ECR](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Elastic-Container-push.png)

Then, tag the latest image which you find on your system with

```bash
pipenv run docker images
```
and push the image.  

Next, go to Elastic Container Service to create a cluster 
![ECS](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS.png)

Select Networking only and go to next step

![ECS1](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(1).png)

Then, choose your cluster name, hit create and then view cluster
![ECS2](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(2).png)
![ECS3](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(3).png)

Now, you need to create a new task and choose Fargate
![ECS4](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(4).png)
![ECS5](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(5).png)

Enter 
* a task definition name
* task role
* Linux as operating system family
* Task memory (0.5 GB)
* Task CPU (0.25 vCPU)

![ECS6](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(6).png)

Then, select add container and choose a container name.
On memory limits select 256 Soft limit and 3000 on Port mappings with tcp.
![ECS7](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(7).png)

For the image you need to have pushed the image and find the URI on your created repo
![imageurl](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Elastic-Container-Registry%20(3).png)

Click add and then create. Go back to clusters,select the created cluster and go to tasks. 

![ECS8](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(8).png)

Choose run new task and select

* Launch type Fargate
* Linux as operating system family
* Cluster VPC and Subnets

![ECS9](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(9).png)

Go to Security groups and add a custom TCP with port 3000 and save

![ECS10](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Amazon-ECS%20(10).png)

Finally, hit run task.

Once the task is running, you will get a public IP.

https://user-images.githubusercontent.com/113017737/200716975-7d780f6e-6d6d-4aeb-ba0e-843f34f240de.mp4


## Further development

I will update this repo with instructions on how to publish on heroku or other service and also create a Streamlit app (frontend) after the project deadline.

### What else can I do?
* Send a pull request.
* If you liked this project, give a ⭐.

Made by [Dimitrios Zacharenakis](https://github.com/dimzachar)

**Connect with me:**

<p align="center">
  <a href="https://www.linkedin.com/in/zacharenakis/" target="blank"><img align="center" src="https://cdn-icons-png.flaticon.com/512/174/174857.png" height="30" width="30" /></a>
  
</p>
           
