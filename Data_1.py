import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot


def Course(df):
    df.replace("?", np.nan, inplace=True)
    print(df)
    print(df.head(5))
    print(df.head(6))
#Evaluation missing values
    missing_data = df.isnull()
#Know, what are variable are missing values and it is quantitiy
    for column in missing_data.columns.values.tolist():
        if missing_data[column].values.sum() != 0:
            print(column)
            print(missing_data[column].values.sum())
    # print (missing_data[column].value_counts())
            print("")
        else:
            pass
#Change the missing values
    avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
    print("Average of normalized-losses:", avg_norm_loss)
    df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
    avg_bore = df["bore"].astype("float").mean(axis=0)
    print("Average of bore:", avg_bore)
    df["bore"].replace(np.nan, avg_bore, inplace=True)
    avg_stroke = df["stroke"].astype("float").mean(axis=0)
    print("Average of stroke:", avg_stroke)
    df["stroke"].replace(np.nan, avg_stroke, inplace=True)
    avg_horsepower = df["horsepower" ].astype("float").mean(axis=0)
    print("Average of horsepower:", avg_horsepower)
    df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)
    avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)
    print("Average of stroke:", avg_peak_rpm)
    df["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace=True)
    print(df['num-of-doors'].value_counts())
# replace the missing 'num-of-doors' values by the most frequent
    print(df['num-of-doors'].value_counts().idxmax())
    print("")
    df["num-of-doors"].replace(np.nan, "four", inplace=True)
# simply drop whole row with NaN in "price" column
    df.dropna(subset=["price"], axis=0, inplace=True)
# reset index, because we droped two rows
    df.reset_index(drop=True, inplace=True)
# Convert data types to proper format
    df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
    df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
    df[["price"]] = df[["price"]].astype("float")
    df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
    print(df.dtypes)
    print("")
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
    df['city-L/100km'] = 235 / df["city-mpg"]
    df['highway-L/100km'] = 235 / df["highway-mpg"]
    print(df.head(5))
    print("")
# replace (original value) by (original value)/(maximum value)
    df['length'] = df['length'] / df['length'].max()
    df['width'] = df['width'] / df['width'].max()
    df['height'] = df['height'] / df['height'].max()
    print(df[["length", "width", "height"]].head())

    print(df.describe(include='all'))
    print(df.info())
    print("")
# Convert data to correct format
    df["horsepower"] = df["horsepower"].astype(int, copy=True)
    plt.pyplot.hist(df["horsepower"])

    # set x/y labels and plot title
    plt.pyplot.xlabel("horsepower")
    plt.pyplot.ylabel("count")
    plt.pyplot.title("horsepower bins")

# We build a bin array, with a minimum value to a maximum value
    bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
    print(bins)
    group_names = ['Low', 'Medium', 'High']
    df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
    print(df[['horsepower', 'horsepower-binned']].head(20))
    print(df["horsepower-binned"].value_counts())
    print(pyplot.bar(group_names, df["horsepower-binned"].value_counts()))
    # set x/y labels and plot title
    plt.pyplot.xlabel("horsepower")
    plt.pyplot.ylabel("count")
    plt.pyplot.title("horsepower bins")
    print("")
    print(df.columns)
    dummy_variable_1 = pd.get_dummies(df["fuel-type"])
    print(dummy_variable_1.head())
    dummy_variable_1.rename(columns={'fuel-type-gas': 'gas', 'fuel-type-diesel': 'diesel'}, inplace=True)
    print(dummy_variable_1.head())
# merge data frame "df" and "dummy_variable_1"
    df = pd.concat([df, dummy_variable_1], axis=1)
# drop original column "fuel-type" from "df"
    df.drop("fuel-type", axis=1, inplace=True)
    print(df.head())
    dummy_variable_2 = pd.get_dummies(df["aspiration"])
    dummy_variable_2.rename(columns={'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
    print(dummy_variable_2.head())
# merge data frame "df" and "dummy_variable_2"
    df = pd.concat([df, dummy_variable_2], axis=1)
# drop original column "aspiration" from "df"
    df.drop("aspiration", axis=1, inplace=True)
    print(df.head())
#   df.to_csv('clean_df.csv')


def main():
    other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
    df = pd.read_csv(other_path, header=None)
    headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
    df.columns = headers
    Course(df)

if __name__ == "__main__":
    main()