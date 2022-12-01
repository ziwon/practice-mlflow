set positional-arguments

venv:
    python3 -m venv .venv && source .venv/bin/activate

deps:
    .venv/bin/pip install -r requirements.txt

mlflow-serve model_uuid='' model='PlainRegressionModel':
    mlflow models serve --model-uri runs:/{{model_uuid}}/{{model}} --env-manager local --port 3020

ui:
    mlflow ui --backend-store-uri mlruns --port 5020

model-train:
    just clean
    python elasticnet_winemodel.py

model-path:
    #!/usr/bin/env bash
    experiment_file=$(ls -td ./mlruns/0/* | head -1)
    model_path=$experiment_file/artifacts/model
    echo $model_path

model-serve:
    #!/usr/bin/env bash
    model_path=$(just model-path)
    cat <<EOF >  model-settings.json
    {
        "name": "wine-classifier",
        "implementation": "mlserver_mlflow.MLflowRuntime",
        "parameters": {
            "uri": "${model_path}"
        }
    }
    EOF
    mlserver start .

model-infer:
    #!/usr/bin/env python3
    import requests
    inference_request = {
        "inputs": [
        {
            "name": "fixed acidity",
            "shape": [1],
            "datatype": "FP32",
            "data": [7.4],
        },
        {
            "name": "volatile acidity",
            "shape": [1],
            "datatype": "FP32",
            "data": [0.7000],
        },
        {
            "name": "citric acid",
            "shape": [1],
            "datatype": "FP32",
            "data": [0],
        },
        {
            "name": "residual sugar",
            "shape": [1],
            "datatype": "FP32",
            "data": [1.9],
        },
        {
            "name": "chlorides",
            "shape": [1],
            "datatype": "FP32",
            "data": [0.076],
        },
        {
            "name": "free sulfur dioxide",
            "shape": [1],
            "datatype": "FP32",
            "data": [11],
        },
        {
            "name": "total sulfur dioxide",
            "shape": [1],
            "datatype": "FP32",
            "data": [34],
        },
        {
            "name": "density",
            "shape": [1],
            "datatype": "FP32",
            "data": [0.9978],
        },
        {
            "name": "pH",
            "shape": [1],
            "datatype": "FP32",
            "data": [3.51],
        },
        {
            "name": "sulphates",
            "shape": [1],
            "datatype": "FP32",
            "data": [0.56],
        },
        {
            "name": "alcohol",
            "shape": [1],
            "datatype": "FP32",
            "data": [9.4],
        },
        ]}
        
    endpoint = "http://localhost:8080/v2/models/wine-classifier/infer"
    response = requests.post(endpoint, json=inference_request)
    
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(response.json())

model-predict:
    #!/usr/bin/env python3
    import requests
    inference_request = {
        "dataframe_split": {
            "columns": [
                "alcohol",
                "chlorides",
                "citric acid",
                "density",
                "fixed acidity",
                "free sulfur dioxide",
                "pH",
                "residual sugar",
                "sulphates",
                "total sulfur dioxide",
                "volatile acidity",
            ],
            "data": [[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]]
        }
    }
    
    endpoint = "http://localhost:8080/invocations"
    response = requests.post(endpoint, json=inference_request)
    
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(response.json())

model-sig:
    #!/usr/bin/env python3
    import requests
    endpoint = "http://localhost:8080/v2/models/wine-classifier"
    response = requests.get(endpoint)
    
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(response.json())

clean:
    rm -rf mlruns
    rm -rf outputs
    rm -rf images

# Local Variables:
# mode: makefile
# End:
# vim: set ft=make :