import mlflow
import os
from random import random, randint
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

print("MLflow Version:", mlflow.version.VERSION)
print("Pandas Version:", pd.__version__)
print("Scikit-learn Version:", sklearn.__version__)
print("Matplotlib Version:", matplotlib.__version__)

# help(mlflow)

def run(run_name=""):
    mlflow.set_experiment("HelloWorld")

    with mlflow.start_run() as r:
        print("Model Run:", r.info.run_uuid)
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("param1", randint(0, 100))

        mlflow.log_metric("foo1", random())
        mlflow.log_metric("foo2", random() + 1)

        mlflow.set_tag("run_origin", "python")

        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        with open("outputs/text.txt", "w") as f:
            f.write("Hello World!")
        
        mlflow.log_artifacts("outputs", artifact_path="artifact")

        mlflow.end_run()

run("HelloWorld-LocalRun")

mlflow.set_tracking_uri("http://localhost:5020")
print("Tracking URI:", mlflow.tracking.get_tracking_uri())