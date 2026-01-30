from pathlib import Path
import pandas as pd
import calendar
from datetime import datetime
from sklearn.model_selection import train_test_split

COLUMNS = [
    'Hotel ID', 'Hotel Name', 'Reviewer', 'Reviewer Profile Link',
    'Review Link', 'Review Title', 'Review Content', 'Overall Rating',
    'Value', 'Location', 'Rooms', 'Service', 'Sleep Quality', 'Cleanliness',
    'City', '# of contributions', '# of helpful votes', 'Travel Type',
    'Manager reply or not?', 'Manager reply content', 'Crawl Time',
    'Stay Year', 'Stay Month', 'Review Year', 'Review Month'
]
RELEVANT_COLUMNS = [
    'Review Title', 'Review Content', 'Overall Rating',
    'Value', 'Location', 'Rooms', 'Service', 'Sleep Quality', 'Cleanliness',
    'Travel Type', 'Stay Year', 'Stay Month', 'Pandemic'
]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess hotel review data.")
    parser.add_argument("--input_file", type=str, default="data/raw/State_Hotel_reviews_v1.txt",
                        help="Path to the raw data file.")
    parser.add_argument("--output_dir", type=str, default="data/processed/",
                        help="Directory to save the processed data.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--eval_size", type=float, default=0.1,
                        help="Proportion of the training dataset to include in the eval split.")
    parser.add_argument("--seed", type=int, default=1207,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

def load_files(file: str):
    # Read files
    data = Path(file).read_text().splitlines()

    # Get columns
    columns = data[0].split("\t")

    # Remove header and filter rows
    data = data[1:]
    data = [row.split("\t") for row in data if len(row.split("\t")) == len(columns)]

    return pd.DataFrame(data, columns=columns)

def filter_empty_data(data: pd.DataFrame):
    filter = data[data["Date of stay(M Y)"] != ""]
    filter = filter[filter["Date of review(Y/M)"] != ""]
    filter = filter[filter["Date of review(Y/M)"] != "None"]
    filter = filter[filter["Overall Rating"] != ""]
    filter = filter[filter["Value"] != ""]
    filter = filter[filter["Location"] != ""]
    filter = filter[filter["Rooms"] != ""]
    filter = filter[filter["Service"] != ""]
    filter = filter[filter["Sleep Quality"] != ""]
    filter = filter[filter["Cleanliness"] != ""]
    filter = filter[filter["Travel Type"] != ""]
    filter = filter[filter["Travel Type"] != "None"]
    return filter

def process_date(data: pd.DataFrame):
    """Split date into year and month."""
    df = data.copy()

    # Month dictionary
    month_to_int = {calendar.month_name[num]: num for num in range(1, 13)}
    int_to_month = {num: calendar.month_name[num] for num in range(1, 13)}

    # Transform `Date of stay(M Y)` to `Month Year`
    def transform_review_date(text):
        if "/" in text:
            return f"{int_to_month[int(text.split('/')[1])]} {text.split('/')[0]}"
        else:
            return f"{text.split(',')[0].split()[0]}{text.split(',')[1]}"
    df["Date of review(Y/M)"] = df["Date of review(Y/M)"].apply(transform_review_date)

    # Split into `Year` and `Month`
    df["Stay Year"] = df["Date of stay(M Y)"].apply(lambda x: x.split(" ")[1])
    df["Stay Month"] = df["Date of stay(M Y)"].apply(lambda x: x.split(" ")[0])
    df["Review Year"] = df["Date of review(Y/M)"].apply(lambda x: x.split(" ")[1])
    df["Review Month"] = df["Date of review(Y/M)"].apply(lambda x: x.split(" ")[0])

    # Convert `Year` and `Month` to `int`
    df["Stay Year"] = df["Stay Year"].astype(int)
    df["Stay Month"] = df["Stay Month"].apply(lambda x: month_to_int[x])
    df["Review Year"] = df["Review Year"].astype(int)
    df["Review Month"] = df["Review Month"].apply(lambda x: month_to_int[x])

    # Drop `Date of stay(M Y)` and `Date of review(Y/M)`
    df = df.drop(columns=["Date of stay(M Y)", "Date of review(Y/M)"])

    return df

def remove_early_data(data: pd.DataFrame):
    """Remove data before 2017-05-01."""
    df = data.copy()
    df["Stay Date"] = df["Stay Year"].astype(str) + "-" + df["Stay Month"].astype(str)
    df["Stay Date"] = pd.to_datetime(df["Stay Date"], format="%Y-%m")
    df = df[df["Stay Date"] >= "2017-05-01"]
    return df

def pandemic_feature(date):
    if datetime.strptime("2017-05-01", "%Y-%m-%d") <= date <= datetime.strptime("2019-12-31", "%Y-%m-%d"):
        return "Pre-Pandemic"
    elif datetime.strptime("2020-01-01", "%Y-%m-%d") <= date <= datetime.strptime("2022-04-30", "%Y-%m-%d"):
        return "During-Pandemic"
    elif datetime.strptime("2022-05-01", "%Y-%m-%d") <= date <= datetime.strptime("2023-05-31", "%Y-%m-%d"):
        return "Post-Pandemic"
    else:
        raise ValueError("Invalid date")

def col_to_int(data: pd.DataFrame):
    for col in ['Overall Rating', 'Value', 'Location', 'Rooms', 'Service', 'Sleep Quality', 'Cleanliness']:
        data[col] = data[col].astype(int)
    return data

def main():
    args = parse_args()
    data = filter_empty_data(args.input_file)
    data = process_date(data)
    data.columns = COLUMNS
    data = remove_early_data(data)
    data["Pandemic"] = data["Stay Date"].apply(pandemic_feature)
    data = data.drop(columns=["Stay Date"])
    data = data.sort_values(by=["Stay Year", "Stay Month"])
    data = col_to_int(data)
    data = data[RELEVANT_COLUMNS]
    data = data.reset_index(drop=True)
    test_size = round(len(data) * args.test_size)
    eval_size = round(len(data) * args.eval_size)
    train, test = train_test_split(data, test_size=test_size, random_state=args.seed)
    train, eval = train_test_split(train, test_size=eval_size, random_state=args.seed)
    data_path = f"{args.output_dir}/hotel_reviews.json"
    train_path = f"{args.output_dir}/train.json"
    eval_path = f"{args.output_dir}/eval.json"
    test_path = f"{args.output_dir}/test.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    data.to_json(data_path, orient="records")
    train.to_json(train_path, orient="records")
    eval.to_json(eval_path, orient="records")
    test.to_json(test_path, orient="records")

if __name__ == "__main__":
    main()