## Capstone Project mlzoomcamp Image Classification 

<p align="center">
<img src="https://github.com/dimzachar/mlzoomcamp_projects/blob/master/01-capstone_project/Images/collage.jpg">
</p>


## Table of Contents
 * [Description of the problem](#description-of-the-problem)
 * [Project Objectives](#project-objectives)
 * [Local deployment](#local-deployment)
 * [Production deployment](#production-deployment-with-bentoml)
   * [Docker container](#docker-container)
   * [Cloud deployment](#cloud-deployment)
 * [Further development](#further-development)
 * [More](#what-else-can-i-do)


Repo contains the following:

* `README.md` with
  * Description of the problem
  * Instructions on how to run the project
* `notebook.ipynb` a Jupyter Notebook with the data analysis and models
* Script `train.py` (suggested name)
  * Training the final model
* Script `lambda-function.py` for predictions. The script is formatted for deployment on Amazon Web Services' Lambda.
* final model .h5
* Files with dependencies
  * `env_project.yml` conda environment (optional)
* Instructions for Production deployment
  * Video or image of how you interact with the deployed service
* Documentation with code description
* The original dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/vencerlanz09/shells-or-pebbles-an-image-classification-dataset)



## Description of the problem

Written with help of #ChatGPT

Have you ever been to the beach and found yourself wanting to collect either shells or pebbles, but not sure which was which? Or maybe you're in the oil and gas industry and need a quick and accurate way to classify different geological materials? Well, I have the solution for you!

Introducing the Shells or Pebbles dataset – a collection of images specifically designed for binary classification tasks. With this dataset, you'll be able to easily determine whether a certain image is a shell or a pebble.

But the usefulness of this dataset doesn't stop there. In the oil and gas industry, accurately identifying and classifying different materials, including rocks and shells, is crucial for exploration and production activities. By understanding the composition and structure of the earth's layers, geologists can make informed decisions about where to drill for oil and gas.

And for those concerned about the environment, this dataset can also be used to study the impacts of climate change on coastal ecosystems. By analyzing the distribution and abundance of shells and pebbles on beaches, scientists can gain valuable insights into the health of marine life and the effects of human activities.

So whether you're an artist looking to create a beach-themed project or a scientist studying the earth's geological makeup, the Shells or Pebbles dataset has something to offer. With its reliable and accurate classification capabilities, this dataset can help you make better informed decisions and better understand the world around you.


## Project Objectives

Potential objectives for this project include:

* Develop a model that performs well on a binary classification problem.
* Tune the model's hyperparameters to get the best possible accuracy.
	* Used learning rate, droprate as main hyperparameters. Also added data augmentation but due to lack of time and computer resources didn't spend much time on tuning it further. Size of inner layers, img size and other parameters could also be changed by the user.
* Use the callbacks to save the best model weights and and end training if the validation accuracy does not increase after a certain number of epochs.
* Utilize TensorBoard to visualize the training process and find trends or patterns in the data. (I didn't make use of this in the end)
* Use the trained model to accurately categorize new photos as Shells or Pebbles.
* Deploy the trained model in a production environment.
* Create comprehensive documentation for the project, including a detailed description of the model architecture, training procedure and deployment.
* Display the project's outcomes in a more professional way.


I selected possible best parameters and architecture to achieve a good accuracy. It is possible the architecture is not much suitable or there are other parameters that better fit this problem. It would need more investigation on the dataset and on creation of the model.

## Local deployment

All development was done on Windows with conda.

You can either recreate my environment by
```bash
conda env create -f env_project.yml
conda activate project
```

or do it on your own environment.

1. Download repo
```bash
git clone https://github.com/dimzachar/mlzoomcamp_projects.git
```

2. For the virtual environment, I utilized pipenv. 

Alternative (optional): You can install all dependencies with `pip` with the following command:

`pip install -r pip-requirements.txt`

> Note: `pip-requirements.txt` was created with the following command:

> `pip list --format=freeze > pip-requirements.txt`




If you want to use the same venv as me, install pipenv and dependencies, navigate to the folder with the given files:

Before you begin you need to download the data. Folder structure should look like this
```
└───data
    └───train
        ├───Pebbles
        └───Shells
```

Once you download the .zip file from Kaggle, create a data folder and inside a train folder and move the 2 folders from the .zip inside the train folder.

```bash
cd 01-capstone_project
pip install pipenv
pipenv install numpy pandas seaborn jupyter plotly scipy tensorflow==2.9.2 scikit-learn==1.1.3 pydantic==1.10.2
```





3. Enter shell. To open the `notebook.ipynb` and see all the models

```bash
pipenv shell
pipenv run jupyter notebook
```

For the following you need to run train.py
```bash
pipenv run python train.py
```





### What else can I do?
* Send a pull request.
* If you liked this project, give a ⭐.

**Connect with me:**

<p align="center">
  <a href="https://www.linkedin.com/in/zacharenakis/" target="blank"><img align="center" src="https://cdn-icons-png.flaticon.com/512/174/174857.png" height="30" width="30" /></a>
  <a href="https://github.com/dimzachar" target="blank"><img align="center" src="https://cdn-icons-png.flaticon.com/512/25/25231.png" height="30" width="30" /></a>

  
</p>
           
