import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pip
from IPython.display import display
from IPython.html import widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def Course(df):
    df.replace("?", np.nan, inplace=True)
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
    print("")
# Convert data to correct format
    df["horsepower"] = df["horsepower"].astype(int, copy=True)
    plt.hist(df["horsepower"])

    # set x/y labels and plot title
    plt.xlabel("horsepower")
    plt.ylabel("count")
    plt.title("horsepower bins")
    plt.show()
# We build a bin array, with a minimum value to a maximum value
    bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
    print(bins)
    print("")
    group_names = ['Low', 'Medium', 'High']
    df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
    print(df[['horsepower', 'horsepower-binned']].head(10))
    print("")
    print(df["horsepower-binned"].value_counts())
    plt.bar(group_names, df["horsepower-binned"].value_counts())
    # set x/y labels and plot title
    plt.xlabel("horsepower")
    plt.ylabel("count")
    plt.title("horsepower bins")
    plt.show()
    print("")
    dummy_variable_1 = pd.get_dummies(df["fuel-type"])
    print(dummy_variable_1.head())
    dummy_variable_1.rename(columns={'fuel-type-gas': 'gas', 'fuel-type-diesel': 'diesel'}, inplace=True)
    print(dummy_variable_1.head())
# merge data frame "df" and "dummy_variable_1"
    df = pd.concat([df, dummy_variable_1], axis=1)
# drop original column "fuel-type" from "df"
    df.drop("fuel-type", axis=1, inplace=True)
    print(df.head())
        #dummy_variable_2 = pd.get_dummies(df["aspiration"])
        #dummy_variable_2.rename(columns={'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
        #print(dummy_variable_2.head())
        #df.drop(['aspiration-std', 'aspiration-turbo'], axis=1, inplace=True)
# merge data frame "df" and "dummy_variable_2"
        #df = pd.concat([df, dummy_variable_2], axis=1)
# drop original column "aspiration" from "df"
    #df.drop("aspiration", axis=1, inplace=True)
        #print(df.head())
    #df.to_csv('Dataset_1.csv')

### ANALYZING INDIVIDUAL FEATURE PATTERN USING VISUALIZATION
    print("")
    print(df.dtypes)
    print("")
    print(df['peak-rpm'].dtypes)
    print("")
# Calculate Correlation between variables float64 and int64
    print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

# Engine size as potential predictor variable of price
    sns.set()
    sns.regplot(x="engine-size", y="price", data=df)
    print("")
    plt.ylim(0,)
    plt.show()
# Examine correlation between engine-size and price
    print(df[['engine-size','price']].corr())
    print("")
    sns.regplot(x="highway-mpg", y="price", data=df)
    plt.show()
    print(df[["highway-mpg","price"]].corr())
    print("")
    sns.regplot(x="peak-rpm", y="price", data=df)
    plt.show()
    print(df[["peak-rpm","price"]].corr())
    print("")
    sns.regplot(x="stroke", y="price", data=df)
    print(df[["stroke","price"]].corr())
    plt.show()
# Categorical variable
    sns.boxplot(x="body-style", y="price", data=df)
    plt.show()
    sns.boxplot(x="engine-location", y="price", data=df)
    plt.show()
    sns.boxplot(x="drive-wheels", y="price", data=df)
    plt.show()
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
    plt.pcolor(grouped_pivot, cmap='RdBu')
    plt.colorbar()
    plt.show()
#### grafica no se como se corre
    fig, ax = plt.subplots()
    im = ax.pcolor(grouped_pivot, cmap='RdBu')

    # label names
    row_labels = grouped_pivot.columns.levels[1]
    col_labels = grouped_pivot.index

    # move ticks and labels to the center
    ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

    # insert labels
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)

    # rotate label if too long
    plt.xticks(rotation=90)

    fig.colorbar(im)
    plt.show()
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

#Conclusion: Important Variable
#Continuous numerical variables:
#Length
#Width
#Curb - weight
#Engine - size
#Horsepower
#City - mpg
#Highway - mpg
#Wheel - base
#Bore

##Categorical variables:
#Drive - wheels
###Linear Regression and Multiple Linear Regression
#Predic the price car with highway-mpg
    lm = LinearRegression()
    X = df[['highway-mpg']]
    Y = df[['price']]
    lm.fit(X,Y)
    Yhat = lm.predict(X)
    print(Yhat[0:5])
    print("")
# Regresition linear Y = AX + B
# B = INTERCEPTOR
# A = COEFICIENT
    print(lm.intercept_)
    print(lm.coef_)
    print('Modelo is the next')
    print("38423.30 -821.73*'highway-mpg'")
    print("")
    lm = LinearRegression()
    lm.fit(df[['engine-size']],df[['price']] )
    Yhat = lm.predict(X)
    print(Yhat[0:5])
    print("")
    print(lm.intercept_)
    print(lm.coef_)
    print('Modelo is the next')
    print("-7963.33 + 166.86*'engine-size'")
    print("")
# MULTIPLE LINEAR REGRESITION
    lm = LinearRegression()
    print(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','width']].corr())
    print("Te amo eriii")
    print("")
    Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
    lm.fit(Z,df[['price']])
    print("")
    print(lm.intercept_)
    print(lm.coef_)
    w = df[['normalized-losses','highway-mpg']]
    lm.fit(w,df[['price']])
    print("")
    print(lm.intercept_)
    print(lm.coef_)
    print("")
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    sns.regplot(x="highway-mpg", y="price", data=df)
    plt.ylim(0, )
    plt.show()
    plt.close()

    plt.figure(figsize=(width, height))
    sns.regplot(x="peak-rpm", y="price", data=df)
    plt.ylim(0, )
    plt.show()
    plt.close()

    print(df[["peak-rpm","highway-mpg","price"]].corr())
    print("")
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    sns.residplot(df['highway-mpg'], df['price'])
    plt.show()
    plt.close()

    lm = LinearRegression()
    #Yhat = lm.predict(Z)
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
    sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)

    plt.title('Actual vs Fitted Values for Price')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

    def PlotPolly(model, independent_variable, dependent_variabble, Name):
        x_new = np.linspace(15, 55, 100)
        y_new = model(x_new)

        plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
        plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
        ax = plt.gca()
        ax.set_facecolor((0.898, 0.898, 0.898))
        fig = plt.gcf()
        plt.xlabel(Name)
        plt.ylabel('Price of Cars')

        plt.show()
        plt.close()

    x = df['highway-mpg']
    y = df['price']

    # Here we use a polynomial of the 3rd order (cubic)
    f = np.polyfit(x, y, 3)
    p = np.poly1d(f)
    print(p)

    PlotPolly(p, x, y, 'highway-mpg')
    np.polyfit(x, y, 3)

    print("")

    f = np.polyfit(x, y, 11)
    p = np.poly1d(f)
    print(p)
    print("")
    PlotPolly(p, x, y, 'Length')
    np.polyfit(x, y, 3)


# We create a PolynomialFeatures object of degree 2:
    pr = PolynomialFeatures(degree=2)
    Z_pr = pr.fit_transform(Z)

# The original data is of 201 samples and 4 features
    print(Z.shape)
    print("")

# after the transformation, there 201 samples and 15 features
    print(Z_pr.shape)
    print("")

# We create the pipeline, by creating a list of tuples including the name of the model or estimator and its corresponding constructor.

# we input the list as an argument to the pipeline constructor
    #pipe=Pipeline(steps=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),('model', LinearRegression())])

# We can normalize the data, perform a transform and fit the model simultaneously.
    #pipe.fit(x_train, y_train)
    #print("")

#  Similarly,  we can normalize the data, perform a transform and produce a prediction  simultaneously
    #ypipe = pipe.predict(Z)
    #print(ypipe[0:4])
    #print("")

### AYUDA HERMANA
# highway_mpg_fi
# Regression with one variable
    lm = LinearRegression()
    lm.fit(X, Y)

# Find the R^2
    print('The R-square is: ', lm.score(X,Y))
    print("We can say that ~ 49.659% of the variation of the price is explained by this simple linear model 'horsepower_fit'.")
    print("")

    Yhat = lm.predict(X)
    print('The output of the first four predicted value is: ', Yhat[0:4])
    print("")

# we compare the predicted results with the actual results
    mse = mean_squared_error(df['price'], Yhat)
    print('The mean square error of price and predicted value is: ', mse)
    print("")

##MULTIPLE LINEAR REGRESSION
    # fit the model
    lm.fit(Z, df['price'])
    # Find the R^2
    print('The R-square is: ', lm.score(Z, df['price']))
    print("We can say that ~ 80.896 % of the variation of price is explained by this multiple linear regression 'multi_fit'")
    print("")

    Y_predict_multifit = lm.predict(Z)
    print('The mean square error of price and predicted value using multifit is: ',mean_squared_error(df['price'], Y_predict_multifit))
    print("")

# Model 3: Polynomial Fit
# Let's calculate the R^2
# let’s import the function r2_score from the module metrics as we are using a different function
    r_squared = r2_score(y, p(x))
    print('The R-square value is: ', r_squared)
    print("")

# Calculate MSE
    mean_squared_error(df['price'], p(x))


# PREDICTION AND DECISION MAKING
    new_input = np.arange(1, 100, 1).reshape(-1, 1)
    lm.fit(X, Y)
    yhat = lm.predict(new_input)
    print(yhat[0:5])
    print("")

    plt.plot(new_input, yhat)
    plt.show()



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
