import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from NYC_TAXI.constants import  OUTPUT_FILE_PATH

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# @click.command()
# @click.option(
#     "--data_path",
#     default="./output",
#     help="Location where the processed NYC taxi trip data was saved"
# )
def run_train():

    X_train, y_train = load_pickle(os.path.join(OUTPUT_FILE_PATH, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(OUTPUT_FILE_PATH, "val.pkl"))

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)


# if __name__ == '__main__':
#     run_train()