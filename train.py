from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from azureml.core.workspace import Workspace
from azureml.data.dataset_factory import TabularDatasetFactory
import numpy as np
import pandas as pd
import joblib
import argparse
import os

def clean_data(data):
    y_df = data.pop("RiskLevel")
    return data, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss', type=str, default='hinge', help="The loss function to be used.")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--alpha', type=float, default=0.0001, help="Constant that multiplies the regularization term.")
    parser.add_argument('--learning_rate', type=str, default='optimal', help="The learning rate schedule.")
    parser.add_argument('--eta0', type=float, default=0.000001, help='The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules.')
    parser.add_argument('--penalty', type=str, default='l2', help='The penalty (aka regularization term) to be used. ')

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Loss function:", str(args.loss))
    run.log("Max iterations:", int(args.max_iter))
    run.log("Alpha:", float(args.alpha))
    run.log("Learning rate:", str(args.learning_rate))
    run.log("eta0:", float(args.eta0))
    run.log("penalty:", str(args.penalty))

    key = "MomHealth Dataset"
    
    #ws = Workspace.from_config()
    # Get the workspace from the run context
    ws = run.experiment.workspace
    ds = ws.datasets[key].to_pandas_dataframe()

    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    from sklearn.model_selection import train_test_split

    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = SGDClassifier(
        loss=args.loss, 
        max_iter=args.max_iter,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        penalty=args.penalty,
        eta0=args.eta0).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model,filename='./outputs/model.joblib')

if __name__ == '__main__':
    main()