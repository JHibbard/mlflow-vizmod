<p align="center">
<img src="https://raw.githubusercontent.com/dennyglee/mlflow-vizmod/main/images/mlflow-vizmod.png" width=280/><br/>
Extending MLflow for Visualization Models
</p>
<br/><br/>

The `mlflow-vizmod` project allows data scientists to be more productive with their visualizations.  We treat visualizations as models - just like ML models - thus being able to use the same infrastructure as MLflow to track, create projects, register, and deploy visualizations.

<p align="center">
<img src="https://raw.githubusercontent.com/dennyglee/mlflow-vizmod/main/images/mlflow-vizmod-flow.png" width=500/>
<br/>
<em>Use the same visualization model to generate and deploy two different visualizations based on different datasets.</em>
</p>

<br/>


We would love to have you try it out and give us feedback through [GitHub Issues](https://github.com/JHibbard/mlflow-vizmod).

Try the `mlflow-vizmod` tutorial using a Jupyter notebook [here]().

## Getting Started

> Note, this is placeholder text until we make the project publicly available.

mlflow-vizmod can be installed using the Python wheel or via `pip`.

```bash
# pip
pip install mlflow-vizmod
```

#### Working with MLflow

```bash
export MLFLOW_TRACKING_URI='http://localhost:5000'
mlflow ui --backend-store-uri sqlite:///mlflow.db
```


## Visualizations are Models
The concept of a model extends far beyond machine learning encompassing any approximation of a system or process we wish to understand.  Machine learning models fit into a category of mathematically rigorous models that can be developed and used with the scientific method (hypothesis testing etc.) but are by no means the beginning or end of the model space continuum.  

Visualizations can be used as models themselves or a means of visually communicating the results, quality etc. of machine learning or other models, to reduce cognitive load on the end user.

* Visualizations require code/concept development and design like any other model this process needs to be captured to be re-usable and reproduceable
* Re-usability and deployability can be enhanced by including the visualization code in a registry, ready to accept compliant data to visualize, much like a machine learning model can be serialized, and deployed, ready to accept new instances of data for inference

This pattern could very effective at orchestrating de-centralized data science workflows, typically found in academia, healthcare, and small startups. It allows developers to push and pull visualizations from a central registry (ie. MLflow deployed with Databricks or as part of a service stack on cloud VMs).


## How to Generate Visualization Model 
To generate a visualization model using `mlflow-vizmod`, use the viz model flavor API which conceptually follows the MLflow API.  For example, the following code snippet is how you log a sklearn model.

```python
# Define and Fit Model
clf = RandomForestClassifier(max_depth=7, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

# Log Accuracy
mlflow.log_metric('accuracy', value=clf.score(X_test, y_test))

# Log Model
mlflow.sklearn.log_model(
    sk_model=clf,
    artifact_path='model',
    signature=infer_signature(X_train, clf.predict(X_train)),
)
```

The following code snippet is how you log a vega lite visualization model.

```python
# Define Viz
viz1 = alt.Chart(
    pd.concat([X_train, y_train], axis=1, sort=False)
).mark_circle(size=60).encode(
    x='sepal length (cm)', y='sepal width (cm)', color='target:N'
).properties(
    height=375, width=575,
).interactive()

# Log Model
mlflow_vismod.log_model(
    model=viz1, 
    artifact_path='viz',
    style='vegalite',
    signature=infer_signature(X_train, None),
    input_example=pd.concat([X_train, y_train], axis=1, sort=False),
)

# Optional: log artifact
viz1.save('./example.html')
mlflow.log_artifact(local_path='./example.html', artifact_path='charts')
```

## How to Load a Visualization Model
Now, that you have defined a model, you can load the saved model and apply it to a different dataset.

```python
# Define Viz
viz2 = alt.Chart(
    pd.concat([X_dtrain, y_dtrain], axis=1, sort=False)
).mark_circle(size=60).encode(
    x='carat', y='clarity', color='cut:N'
).properties(
    height=375, width=575,
).interactive()

# Load model
model_uri = os.path.join(run1.to_dictionary()['info']['artifact_uri'], 'viz')
loaded = mlflow_vismod.load_model(
    model_uri=model_uri,
    style='vegalite'
)

# Rendering new data / subset of data
new_data = pd.concat([X_train, y_train], axis=1, sort=False)
loaded.display(model_input=new_data[new_data['target'] != 2])
```

With `mlflow-vizmod` you can create a new visualization using the same model from the first visualization (i.e. `model_uri`).

<img src="https://raw.githubusercontent.com/dennyglee/mlflow-vizmod/main/images/mlflow-vizmod_2-chart-1-model.gif" width=1000/>


## Deploy an Interactive Web Visualization with MLflow Registry
You can deploy an interactive web visualization using `mlflow-vizmod`.  This use case is particuarly effective with Vegalite, since models are encoded as JSON objects, which accept pointers to data. This means you can store the JSON object as an artifact and use the visualization API to point it at new data.

<p align="center">
<img src="https://raw.githubusercontent.com/dennyglee/mlflow-vizmod/main/images/mlflow-vizmod-deploy-flow.png" width=700 align="center"/>
</p>

To do this:
* Encode a visualization (ie. a geo map for ICU bed utilization and covid cases for each county in the USA with size/color encodings and tooltips which help the end user make decisions on where patients or resources should be sent)
* Register the visualization
* Pull the visualization and point it at a compliant data set of interest
* Deploy to a website (the website would already have panel or some other mechanism of accpeting the registered visualization)
* Alter the visualization as needed, using registry tags to point the new version of the model


<img src="https://raw.githubusercontent.com/dennyglee/mlflow-vizmod/main/images/mlflow-vizmod_model-repository.gif" width=1000/>



## Action Items

* Update notebook so instead of logging viz2, load the model saved in viz1 and create the visualization based on the model saved in viz1.
* Add another code example of deploying visualization model to MLflow Model Registry and then load the model.





