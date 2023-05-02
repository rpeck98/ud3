from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
import numpy as np
import pandas as pd
import argparse
import os
import joblib

def clean_data(data):
    y_df = data.pop("RiskLevel")
    return data, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--penalty', type=str, default='l2', help="Penalty function")
    parser.add_argument('--multi_class', type=str, default='auto', help="Type of multi class to perform")
    parser.add_argument('--solver', type=str, default='auto', help="Solver to use with logistic regression")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))
    run.log("Penalty:", str(args.penalty))
    run.log("Multi class:", str(args.multi_class))
    run.log("Solver:", str(args.solver))

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

    model = LogisticRegression(
        C=args.C, 
        max_iter=args.max_iter,
        penalty=args.penalty,
        multi_class=args.multi_class,
        solver=args.solver).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model,filename='./outputs/model.joblib')

if __name__ == '__main__':
    main()