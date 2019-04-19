import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
import seaborn as sns
from scipy import stats

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

### ANALYZING INDIVIDUAL FEATURE PATTERN USING VISUALIZATION
    print("")
    print(df.dtypes)
    print(df['peak-rpm'].dtypes)
    print("")
# Calculate Correlation between variables float64 and int64
    print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

# Engine size as potential predictor variable of price
    sns.set()
    sns.regplot(x="engine-size", y="price", data=df)
    print("")
    #plt.ylim(0,0)
# Examine correlation between engine-size and price
    print(df[['engine-size','price']].corr())
    print("")
    sns.regplot(x="highway-mpg", y="price", data=df)
    print(df[["highway-mpg","price"]].corr())
    print("")
    sns.regplot(x="peak-rpm", y="price", data=df)
    print(df[["peak-rpm","price"]].corr())
    print("")
    sns.regplot(x="stroke", y="price", data=df)
    print(df[["stroke","price"]].corr())
# Categorical variable
    sns.boxplot(x="body-style", y="price", data=df)
    sns.boxplot(x="engine-location", y="price", data=df)
    sns.boxplot(x="drive-wheels", y="price", data=df)
    print("")
### DESCRITIVE STATISTICAL ANALYSIS
    print(df.describe(include=['object']))
# How many object are in the variable
    print(df['drive-wheels'].value_counts())
# with the name of the variabel
    print(df['drive-wheels'].value_counts().to_frame())
# Repeat steps but with data frame called drive_wheels_counts
    drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
    drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
# Rename name index
    drive_wheels_counts.index.name = 'drive-wheels'
    print(drive_wheels_counts)
    print("")
    engine_loc_counts = df['engine-location'].value_counts().to_frame()
    engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
    engine_loc_counts.index.name = 'engine-location'
    print(engine_loc_counts)
    print("")

### BASIC OF GROUPING

    print(df['drive-wheels'].unique())
    print("")
    print(df['body-style'].unique())
    print("")
    df_group_one = df[['drive-wheels','body-style','price']]
# Get mean with te variable object (drive-wheels)
    df_group_one1 = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
    df_group_one2 = df_group_one.groupby(['body-style'], as_index=False).mean()
    print(df_group_one1)
    print("")
    print(df_group_one2)
    print("")
# Get mean with two variable object
    df_gptest = df[['drive-wheels', 'body-style', 'price']]
    grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    print(grouped_test1)
    print("")
# With the excel
    grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
    print(grouped_pivot)
    print("")
# Fill NaN with 0,but can anything value
    grouped_pivot = grouped_pivot.fillna(0)  # fill missing values with 0
    print(grouped_pivot)
    print("")
#use the grouped results
    #plt.pyplot.figure()
    #plt.pcolor(grouped_pivot, cmaps='RdBu')
    #plt.colorbar()
    #plt.show()
#### grafica no se como se corre
    pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    print("")
    pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
    print("")
    pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
    print("")
    pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    print("")
    pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    print("")
    pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    print("")
    pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    print("")
    pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    print("")
    pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
    print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
    print("")

#### ANOVA
#We can obtain the values of the method group using the method "get_group" with index
    grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
    print(grouped_test2.get_group('4wd')['price'])
    print("")
    f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
                                  grouped_test2.get_group('4wd')['price'])
    print("ANOVA results: F=", f_val, ", P =", p_val)
    print("")
#Separately: fwd and rwd
    f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])
    print("ANOVA results: F=", f_val, ", P =", p_val)
    print("")
#Between 4wd and rwd
    f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])
    print("ANOVA results: F=", f_val, ", P =", p_val)
    print("")
#Between fwd and 4wd
    f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('4wd')['price'])
    print("ANOVA results: F=", f_val, ", P =", p_val)
    print("")

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
