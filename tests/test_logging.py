# Standard Libraries
import os

# External Libraries
import mlflow
import altair as alt
import pandas as pd

# Internal Libraries
import mlflow_vismod


def get_latest_run():
    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id='0')[0].run_id)


def test_logging():
    df = pd.DataFrame({
        'x': [0, 1, 2, 3, 4, ],
        'y': [0, 1, 2, 3, 4, ]
    })

    viz = alt.Chart(
        df
    ).mark_circle(size=60).encode(
        x='x',
        y='y',
    )

    mlflow_vismod.log_model(
        model=viz,
        artifact_path='viz',
        style='vegalite',
    )


def test_loading():
    run = get_latest_run()
    viz = mlflow_vismod.load_model(
        model_uri=os.path.join(
            run.info.artifact_uri,
            'viz',
        ),
        style='vegalite',
    )


def test_display():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, ],
        'y': [1, 2, 3, 4, 5, ]
    })

    run = get_latest_run()
    viz = mlflow_vismod.load_model(
        model_uri=os.path.join(
            run.info.artifact_uri,
            'viz',
        ),
        style='vegalite',
    )
    viz.display(df)
