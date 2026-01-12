import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_numerical(col, df, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[col], kde=True, ax=ax[0])
    sns.boxplot(x=df[col], ax=ax[1])
    fig.suptitle(title)
    return fig


def plot_categorical(col, df, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    df[col].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(title)
    return fig


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax, annot=True)
    return fig
