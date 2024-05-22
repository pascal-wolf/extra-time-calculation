# Automated calculation of extra time in football games using CNN Classification

Intentional time-wasting and incorrect referee assessments can lead to shortened injury times in football, which can disadvantage clubs and cause economic damage.

![Model Prediction Example](/images/foul_example_one.png)
![Explainability Example](/images/foul_explainability_two.png)

As the videos are too large too upload on git, feel free to contact us here in case you are interested to get the demo video.

### Objective
The aim of this work is to develop an appropriate algorithm using historical data and computer vision methods to predict the recommended injury time through automated detection.

### Method
In the course of this work, a CNN model with subsequent binary classification will be used to predict whether play is happening in the current frame of the video or not. By summing up the predictions for each half, the injury time can then be indicated. We also intend to get an understanding based on which pixel the CNN model makes its prediction, as shown in one screenshot of our demo video above.


## Dataset
12 videos of half-times (Bundesliga games) each with 45 minutes of video material will be used. Data from a Kaggle Challenge by the DFL will be used. These will be manually labeled. [Link](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data) to the data.

## Usage

To use the project, make sure you have Python version 3.10 or above installed. 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all requirements.
We recommend using a virtual environment before installing the requirements.

```bash
pip install -r requirements.txt
```

To run the classification model on existing data execute the following command in your terminal.

```bash
python testing.py
```

If you want to train a new model you have to run the training script, which uses MLFlow for the model tracking.

```bash
python train_mlflow.py
```


## Authors

- [@pascal-wolf](https://github.com/pascal-wolf)
- [@nicolas-wolf](https://github.com/nicolas-wolf)