from src.SchoolID_Prediction.code_dir.main import ToReturnModel
from pathlib import Path

if __name__ == "__main__":
    t = ToReturnModel()
    csv_path = Path("src/SchoolID_Prediction/raw_csv_dir/ExamData.csv")
    save_model_dir_path = Path("src/SchoolID_Prediction/all_models")
    save_csv_dir_path = Path("src/SchoolID_Prediction/all_csv")
    json_file_path = Path("scores.json")
    out_json_file_path = Path("output.json")

    # DurationID=1:18 data
    # DurationID=2:7 data
    # DurationID=3:26 Data
    # DurationId=4:72 data
    final_dict = dict()
    for duration_id, no_of_times_to_run in zip(range(1, 5), [18, 7, 26, 72]):
        t.combine_all(csv_path, save_model_dir_path, save_csv_dir_path, json_file_path)
        li = t.predict_data(
            model_dir_path=Path(save_model_dir_path),
            duration_id=duration_id,
            no_of_times_to_run=no_of_times_to_run,
        )
        final_dict.update({f"DurationID_{duration_id}": li})
    t.write_json(json_file_path=out_json_file_path, content=final_dict)
    print(final_dict)
