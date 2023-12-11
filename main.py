import os
import openai
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema
from dotenv import load_dotenv

load_dotenv()
openai_key = os.environ.get('OPENAI_API_KEY')


if openai_key is not None:
    print("OpenAI Key found!")
    openai.api_key = openai_key
else:
    print("OpenAI Key not found. Please set the OPENAI_KEY environment variable.")

mlflow.set_tracking_uri("http://127.0.0.1:5000/")


with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="text-davinci-002",
        task=openai.Completion,
        artifact_path="model",
        prompt="Clasify the following tweet's sentiment: '{tweet}'.",
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(["I believe in a better world"]))


"""
# ******************************************************************************
# Completions using inference parameters
# ******************************************************************************
"""

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="text-davinci-002",
        task=openai.Completion,
        artifact_path="model",
        prompt="Clasify the following tweet's sentiment: '{tweet}'.",
        signature=ModelSignature(
            inputs=Schema([ColSpec(type="string", name=None)]),
            outputs=Schema([ColSpec(type="string", name=None)]),
            params=ParamSchema(
                [
                    ParamSpec(name="max_tokens", default=16, dtype="long"),
                    ParamSpec(name="temperature", default=0, dtype="float"),
                    ParamSpec(name="best_of", default=1, dtype="long"),
                ]
            ),
        ),
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(["I believe in a better world"], params={"temperature": 1, "best_of": 5}))