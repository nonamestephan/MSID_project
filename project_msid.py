import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


def get_dataset(dataset):
    columns = dataset.columns.values
    label_encoder = LabelEncoder()
    # onehot_encoder = OneHotEncoder(sparse=False)
    for column in columns:
        if dataset[column].dtype != np.int64 and dataset[column].dtype != np.float64:
            res = label_encoder.fit_transform(dataset[column])
            dataset[column] = res
    return dataset


def main():
    dataset = pd.read_csv('videogames.csv')
    dataset = get_dataset(dataset)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 14].values

    # print(X)

    X = StandardScaler().fit_transform(X)

    classifiers = {
        "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

    # print(X_train)
    # print(y_train)

    for classifier_idx, (classifier_name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)

        print(classifier_name)
        conf_mat = confusion_matrix(y_test, y_predict)
        ax = sns.heatmap(conf_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
        ax.set_title(classifier_name + '\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')

        plt.tight_layout()
        plt.show()

        clf_report = classification_report(y_test, y_predict, output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)

        plt.tight_layout()
        plt.show()
        print(confusion_matrix(y_test, y_predict))
        print(classification_report(y_test, y_predict))


if __name__ == '__main__':
    main()
