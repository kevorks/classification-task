import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import logging.config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# load logging config
logging.config.fileConfig("logger.conf")

# Folder and the CSV file
data_folder = '../data'
csv_filename = 'titanic.csv'

# Get the current working directory
current_directory = os.getcwd()

# The full path to the CSV file
csv_path = os.path.join(current_directory, data_folder, csv_filename)

if __name__ == '__main__':
    logging.info('Process started')

    # Read the CSV file into a DataFrame
    logging.info('Loading the data')
    df = pd.read_csv(csv_path)

    # Get a data snapshot
    logging.info('Data snapshot')
    print(df)

    # Drop rows with missing values in the 'Survived'
    # column and convert the feature to integer
    df.dropna(subset=['Survived'], inplace=True)
    df['Survived'] = df['Survived'].astype(int)

    df['Cabin_section'] = df['Cabin'].str[0]

    # Get relevant plots
    logging.info('Getting relevant explanatory plots')

    # Count the number of survivors
    survived_counts = df['Survived'].value_counts()

    # Create a bar plot for the target feature
    plt.bar(survived_counts.index, survived_counts.values)
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.title('Survival Frequency (1 = Survived, 0 = Not Survived)')
    plt.savefig('../figs/survived_frequency.png')

    # Count the number of survivors and non-survivors by gender
    survival_gender_counts = df.groupby(['Sex', 'Survived']).size().unstack()

    # Create a bar plot
    survival_gender_counts.plot(kind='bar', stacked=True)
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Survival Frequency by Gender')
    plt.legend(['Not Survived', 'Survived'], loc='upper left')
    plt.xticks(rotation=0)
    plt.savefig('../figs/survived_gender.png')

    # Create a contingency table (cross-tabulation) between
    # 'Cabin_section', 'Pclass', and 'Survived'
    contingency_table = pd.crosstab(
        index=df['Cabin_section'],
        columns=[df['Pclass'],
                 df['Survived']],
        values=df['Survived'],
        aggfunc='count'
    )

    # Plot a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table,
                annot=True,
                cmap='YlGnBu',
                fmt='g',
                linewidths=0.5)

    # Set the title and labels
    plt.title('Survival Count by Cabin Section and Pclass')
    plt.xlabel('Pclass and Survived (0 = No, 1 = Yes)')
    plt.ylabel('Cabin Section')
    plt.savefig('../figs/heatmap_cabin_pclass_survived.png')

    # Data preprocessing
    features = [
        'Survived',
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Cabin_section',
        'Embarked'
    ]

    df = df[features]

    logging.info(
        f'Data preprocessing. Selected features: {", ".join(features)}')

    # Identify numerical and categorical features
    numerical_features = df.select_dtypes(
        include=[np.number, 'float']).columns.tolist()

    categorical_features = df.select_dtypes(
        include=['object']).columns.tolist()

    # Impute missing values for numerical features with the median
    for feature in features:
        if feature in numerical_features:
            median_value = df[feature].median()
            df[feature].fillna(median_value, inplace=True)
            logging.info(
                f'Numerical features "{feature}" imputed with median.')

    # Impute missing values for categorical features with the mode
    for feature in features:
        if feature in categorical_features:
            most_frequent_value = df[feature].mode().values[0]
            df[feature].fillna(most_frequent_value, inplace=True)
            logging.info(
                f'Categorical features "{feature}" imputed with mode.')

    # Check for missing values in the DataFrame
    missing_values = df.isnull().sum()

    if missing_values.sum() == 0:
        logging.info('Missing values in DataFrame: False')
    else:
        logging.info(f'Missing values in DataFrame:\n{missing_values}')

    # Build the model
    logging.info('Building the Model')

    # Separate categorical and numerical features
    categorical_features = ['Sex', 'Cabin_section', 'Embarked']
    numerical_features = [
        col for col in df.columns if col not in
        categorical_features and col != 'Survived']

    # Apply one-hot encoding to categorical features
    logging.info('One-hot encoding for categorical features')
    df_encoded = pd.get_dummies(
        df, columns=categorical_features, drop_first=True
    )

    # Splitting the DataFrame
    X = df_encoded.drop(columns=['Survived'])
    y = df_encoded['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )

    logging.info(f'Sahpe of train dataset:\n{X_train.shape}')
    logging.info(f'Shape of test dataset:\n{X_test.shape}')

    # Initialize the models
    logging.info('Fitting the Models')
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1),
        'LightGBM': LGBMClassifier(objective='binary', n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_jobs=-1)
    }

    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        logging.info(f'Training {model_name}')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f'{model_name} Accuracy: {accuracy}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name

    logging.info(f'Best Model: {best_model} with Accuracy: {best_accuracy}')

    # Evaluate the best model
    logging.info('Model Evaluation Results')
    best_model = models[best_model]
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    logging.info('Process ended successfully')
