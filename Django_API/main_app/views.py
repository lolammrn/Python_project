import matplotlib.pyplot as plt
from django.shortcuts import render

import base64
import seaborn as sns
import numpy as np
from io import BytesIO
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


def get_graph_img():
    buf1 = BytesIO()
    plt.savefig(
        buf1, format="png", bbox_inches='tight')
    graph = base64.b64encode(buf1.getvalue()).decode("utf-8")
    buf1.close()
    return graph


def index(request):
    plt.clf()
    x_graph2 = request.GET.get('x_graph2', 'Age')
    bootstrap= request.GET.get('bootstrap', True)
    max_depth= int(request.GET.get('max_depth', "90"))
    max_features= int(request.GET.get('max_features', "10"))
    min_samples_leaf= int(request.GET.get('min_samples_leaf', "3"))
    min_samples_split= int(request.GET.get('min_samples_split', "3"))
    n_estimators= int(request.GET.get('n_estimators', "200"))

    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    df_viz = df.groupby(['NObeyesdad']).size()

    # graph1
    df_viz.plot(kind='bar')
    graph1 = get_graph_img()

    # graph2
    fig, ax = plt.subplots()
    ax.scatter(df[x_graph2], df["Weight"], c="green",
               alpha=0.5, marker=".", label="Individual")
    ax.set_xlabel(x_graph2)
    ax.set_ylabel("Weight")
    ax.legend()
    graph2 = get_graph_img()

    # graph3
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    locs, labels = plt.xticks()

    fig.suptitle('categorical variables distribution')
    sns.countplot(ax=axes[0,0], data=df,x='Gender')
    sns.countplot(ax=axes[0,1], data=df,x='family_history_with_overweight')
    sns.countplot(ax=axes[0,2], data=df,x='FAVC')
    sns.countplot(ax=axes[0,3], data=df,x='SCC')
    sns.countplot(ax=axes[1,0], data=df,x='MTRANS')
    sns.countplot(ax=axes[1,1], data=df,x='SMOKE')
    sns.countplot(ax=axes[1,2], data=df,x='CAEC')
    sns.countplot(ax=axes[1,3], data=df,x='CALC')  

    axes[1,0].tick_params(labelrotation=45)
    axes[1,2].tick_params(labelrotation=45)
    axes[1,3].tick_params(labelrotation=45)

    graph3 = get_graph_img()



    #randomForest algorithm

    encoder = OneHotEncoder(sparse=False)
    encoder_df = pd.DataFrame(encoder.fit_transform(df[['MTRANS']]))
    encoder_df.columns = encoder.get_feature_names(['MTRANS'])
    df = df.drop('MTRANS', axis=1)
    df = pd.concat([df, encoder_df], axis=1)
    label_encoder = LabelEncoder()
    label_encoder.fit(df['CAEC'])
    df['CAEC'] = label_encoder.transform(df['CAEC'])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['CALC'])
    df['CALC'] = label_encoder.transform(df['CALC'])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Gender'])
    df['Gender'] = label_encoder.transform(df['Gender'])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['family_history_with_overweight'])
    df['family_history_with_overweight'] = label_encoder.transform(df['family_history_with_overweight'])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['FAVC'])
    df['FAVC'] = label_encoder.transform(df['FAVC'])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['SMOKE'])
    df['SMOKE'] = label_encoder.transform(df['SMOKE'])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['SCC'])
    df['SCC'] = label_encoder.transform(df['SCC'])

    #split data into X and y
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X_train) # fit only on training data
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)  # apply same transformation to test data

    model = RandomForestClassifier(bootstrap=bootstrap, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracyScore = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    confusionMatrix = pd.DataFrame(confusionMatrix).to_html()
    classificationReport = classification_report(y_test, y_pred)

    context = {'graph1': graph1, 'graph2': graph2, 'graph3': graph3, "accuracyScore": accuracyScore, "confusionMatrix": confusionMatrix, "classificationReport": classificationReport,
        "bootstrap": bootstrap, "max_depth": max_depth, "max_features": max_features, "min_samples_leaf": min_samples_leaf, "min_samples_split": min_samples_split, "n_estimators": n_estimators
    }
    return render(request, 'main_app/index.html', context)
