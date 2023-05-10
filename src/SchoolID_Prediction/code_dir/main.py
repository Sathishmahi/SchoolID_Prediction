import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
from sklearn.model_selection import GridSearchCV
from json.decoder import JSONDecodeError
from SchoolID_Prediction import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle

duration_idcolumn_name = "DurationID"
out_column_name = "SchoolID"
date_columns_name = "TDate"
duration_id_range = 5


class ToReturnModel:
    def __init__(self) -> None:
        pass

    def to_read_csv(self, csv_path: Path) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError("file not found")
        else:
            try:
                df = pd.read_csv(csv_path)
                return df
            except Exception as e:
                raise e

    def to_split_train_test(
        self, x: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.15
    ):
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=100
            )
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise e

    def to_split_data_based_duration_id(
        self, full_df: pd.DataFrame, duration_id: int
    ) -> pd.DataFrame:
        try:
            return full_df[full_df[duration_idcolumn_name] == duration_id]
        except KeyError as e:
            raise KeyError(f"column not found {duration_idcolumn_name}")
        except Exception as e:
            raise e

    @staticmethod
    def save_model(model, model_path: Path, log=True):
        try:
            with open(model_path, "wb") as pickle_file:
                pickle.dump(model, pickle_file)
                if log:
                    logging.info(f"succesfully model saved {model_path}")

        except Exception as e:
            logging.exception(f"error occured {e}")
            raise OSError(f"root_dir not found {os.path.split(model_path)[0]}")

        except Exception as e:
            logging.exception(f"error occured {e}")
            raise e

    @staticmethod
    def load_model(model_path: Path):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"pickle file not found {model_path}")
            with open(model_path, "rb") as pickle_file:
                model = pickle.load(pickle_file)
            return model
        except Exception as e:
            raise e

    @staticmethod
    def to_check_dtypes(df: pd.DataFrame, convert_dtype: str = "int16") -> pd.DataFrame:
        try:
            column_name_with_dtype = {
                column: convert_dtype
                for column in df.columns
                if df[column].dtypes in ["int", "float"]
            }
            return df.astype(column_name_with_dtype)
        except Exception as e:
            raise e

    @staticmethod
    def to_save_csv(df: pd.DataFrame, file_path: Path, log=True) -> None:
        try:
            if log:
                logging.info(msg=f"successfully csv file saved {file_path}")
            df.to_csv(file_path, index=False)
        except OSError as e:
            logging.exception(msg=f"error occured {e}")
            raise OSError(f"file root_dirnot found {os.path.split(file_path)[0]}")
        except Exception as e:
            logging.exception(msg=f"error occured {e}")
            raise e

    def predict_data(
        self,
        model_dir_path: Path,
        duration_id: int,
        no_of_times_to_run: int,
        data_point: str = "7-5-2023",
    ) -> list:
        if duration_id > 0 and duration_id <= 4:
            try:
                model_name = f"{duration_idcolumn_name}_{duration_id}.pkl"
                model_path = os.path.join(model_dir_path, model_name)
                with open(model_path, "rb") as pickle_file:
                    model = pickle.load(pickle_file)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"file or dir not found {model_path}")
            data_point_split = data_point.split("-")
            predicted_out_list = []
            for no in range(no_of_times_to_run):
                data_point = [
                    [
                        1,
                        int(data_point_split[-1]),
                        int(data_point_split[1]),
                        int(data_point_split[0]),
                        no,
                    ]
                ]
                predicted_out_list.append(model.predict(data_point)[0])
            predicted_out_list = [
                int(abs(predict_data - random.randint(10, 20)))
                for predict_data in predicted_out_list
            ]
            return predicted_out_list
        else:
            raise Exception(f"duration id range between(1,4) you pass {duration_id}")

    @staticmethod
    def to_split_x_and_y(df: pd.DataFrame) -> tuple:
        try:
            x = df.drop(columns=[date_columns_name, duration_idcolumn_name, out_column_name])
            y = df[out_column_name]
            return x, y
        except KeyError as e:
            raise KeyError(f"column not found in axis")
        except Exception as e:
            raise e

    @staticmethod
    def model_training(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        # models=[RandomForestRegressor(),KNeighborsRegressor(),LGBMRegressor(),GradientBoostingRegressor(),DecisionTreeRegressor(),XGBRFRegressor()]
        try:
            random_forest = RandomForestRegressor()
            random_forest.fit(x_train, y_train)
            y_pre = random_forest.predict(x_test)
            score = random_forest.score(x_test, y_test)
            return random_forest, score
        except Exception as e:
            raise e

    @staticmethod
    def _json_helper(json_file_path: Path, content: dict):
        with open(json_file_path, "w") as json_file:
            json.dump(content, json_file)

    @staticmethod
    def read_json(json_file_path: Path) -> dict:
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"jsonm file not found {json_file_path}")
        with open(json_file_path, "r") as json_file:
            try:
                return json.load(json_file)
            except JSONDecodeError as e:
                raise Exception(f"json file empty {json_file_path}")

    def write_json(self, json_file_path: Path, content: dict):
        if not os.path.exists(json_file_path):
            self._json_helper(json_file_path=json_file_path, content=content)

        else:
            if os.path.exists(json_file_path) and not os.path.getsize(json_file_path):
                self._json_helper(json_file_path=json_file_path, content=content)
            else:
                already_existing_content = self.read_json(json_file_path=json_file_path)
                [
                    already_existing_content.update({key: value})
                    for key, value in content.items()
                ]
                self._json_helper(
                    json_file_path=json_file_path, content=already_existing_content
                )

    @staticmethod
    def to_add_hour(df_duration: pd.DataFrame) -> pd.DataFrame:
        df_duration["year"] = df_duration[date_columns_name].dt.year
        df_duration["day"] = df_duration[date_columns_name].dt.day
        df_duration["month"] = df_duration[date_columns_name].dt.month
        test_df = pd.DataFrame()
        dataset_dict = dict()
        for group, data in df_duration.groupby("month"):
            data["hour"] = list(range(1, len(data) + 1))
            test_df = pd.concat((test_df, data))
        return test_df

    def combine_all(
        self,
        csv_path: Path,
        save_model_dir_path: Path,
        save_csv_dir_path: Path,
        json_file_path: Path,
    ):
        try:
            score_dict = dict()
            df = self.to_read_csv(csv_path=csv_path)

            df[date_columns_name] = pd.to_datetime(df[date_columns_name])
            for duration_id in range(1, duration_id_range):
                duration_df = self.to_split_data_based_duration_id(
                    full_df=df, duration_id=duration_id
                )
                duration_df = self.to_check_dtypes(df=duration_df)
                csv_file_path = os.path.join(
                    save_csv_dir_path, f"{duration_idcolumn_name}_{duration_id}.csv"
                )
                self.to_save_csv(df=duration_df, file_path=csv_file_path)
                duration_df = self.to_add_hour(df_duration=duration_df)
                x, y = self.to_split_x_and_y(df=duration_df)
                x_train, x_test, y_train, y_test = self.to_split_train_test(x=x, y=y)
                model, score = self.model_training(
                    x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
                )
                model_path = os.path.join(
                    save_model_dir_path, f"{duration_idcolumn_name}_{duration_id}.pkl"
                )
                self.save_model(model=model, model_path=model_path)
                score_dict.update({f"{duration_idcolumn_name}_{duration_id}": score})
            self.write_json(json_file_path=json_file_path, content=score_dict)

        except Exception as e:
            raise e
