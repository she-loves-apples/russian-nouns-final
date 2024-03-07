"""
    Make sure these libraries and jupyter notebook are installed on your system.
    See the requirements.txt file for a list of dependecies. You can install them with "pip install -r requirements.txt"
"""
import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

list_forms = ["nom_sg", "gen_sg", "dat_sg", "acc_sg", "inst_sg", "prep_sg", "nom_pl", "gen_pl", "dat_pl", "acc_pl",
              "inst_pl", "prep_pl"]


def get_categories(df):
    categories = []
    category_labels = []
    stems_list = list(df.index)
    list_len = len(stems_list)
    df_only_group_col = df.drop(
        ["nom_sg", "gen_sg", "dat_sg", "acc_sg", "inst_sg", "prep_sg", "nom_pl", "gen_pl", "dat_pl", "acc_pl",
         "inst_pl", "prep_pl", "gender"], axis=1)
    df_to_nparray = df_only_group_col.to_numpy()
    categories_list_nparray = np.asarray(df_only_group_col.columns.values.tolist())
    for i in range(0, list_len):
        cat_condition = (df_to_nparray[i, :].astype(float) == 1)
        cat = categories_list_nparray[cat_condition]
        if cat.size == 0:
            print(df_to_nparray[i, :])
            print(categories_list_nparray)
            print(cat)
            print(cat_condition)
            print(i)
            print(list_len)
        categories = np.append(categories, cat[0])
        category_labels = np.append(category_labels, ' '.join(cat))
    return categories, category_labels


# Calculating the average vector for each word with the help of np.mean()
# base form
def average_vec(df, ft):
    average_vector = []
    for index, row in df.iterrows():
        # Saving all word forms of each word to noun_forms
        noun_forms = row[list_forms].tolist()
        # Getting word vectors for each word form
        word_form_vectors = [ft.get_word_vector(x) for x in noun_forms]
        # Calculating mean using numpy.mean()
        word_form_mean = np.mean(word_form_vectors, axis=0)
        # Saving the results in the list
        average_vector = np.append(average_vector, word_form_mean)
    average_vector_array = np.array(average_vector).reshape((len(average_vector) // 300, 300))
    return average_vector_array


# Function that calculates the difference between two arguments
def diff_cal(column_name, average_vectors_list, ft):
    result_diff = np.subtract([ft.get_word_vector(x) for x in column_name], average_vectors_list)
    return result_diff


# Function for creating a time stamp
def datestring():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H-%M-%S")
    return dt_string


# PCA with n_component=50
def run_pca(vectors):
    # Calculating PCA
    pca_50 = PCA(n_components=50)
    pca = pca_50.fit_transform(vectors)
    return pca


# t-SNE after PCA
def run_pca_tsne(vectors):
    # Choosing two principal components as we want to have two-dimensional plot
    tsne = TSNE(n_components=2, perplexity=50)
    # Calculating PCA
    pca_50 = PCA(n_components=50)
    """
        fit_transform: fit the model with X and apply the dimensionality reduction on X; returns transformed values
        X(training data): array-like of shape (n_samples, n_features) where n_samples is the number of samples 
        and n_features is the number of features
    """
    pca_result_50 = pca_50.fit_transform(vectors)
    pca_tsne = tsne.fit_transform(pca_result_50)
    return pca_tsne


def save_html_plots_to_directory(plot, subdirectory_name):
    plot_directory = os.path.join("plotly_interactive_plots", "big_groups", subdirectory_name)
    plot_path = os.path.join(plot_directory, f"plot-{datestring()}.html")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot.write_html(plot_path)


# Visualization of the results after t-SNE
def map_tsne(res_pca_tsne, categories, data_frame, category_labels):
    df = pd.DataFrame()
    df["comp-1"] = res_pca_tsne[:, 0]
    df["comp-2"] = res_pca_tsne[:, 1]
    df["y"] = categories
    df["word"] = list(data_frame.index)
    df["label"] = category_labels
    fig = px.scatter(df, x="comp-1", y="comp-2", custom_data=["word", "label"], color=df.y.tolist())
    fig.update_traces(
        hovertemplate="<br>".join([
            "ColX: %{customdata[0]}",
            "Col1: %{customdata[1]}"
        ])
    )
    # Info from https://plotly.com/python/reference/layout/# layout-paper_bgcolor
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    fig.show()
    # Saving an interactive plot to the directory in the project
    save_html_plots_to_directory(fig, "tsne_results")


def map_tsne_one_group(res_pca_tsne, df_group):
    cat_res = get_categories(df_group)
    df = pd.DataFrame()
    df["comp-1"] = res_pca_tsne[:, 0]
    df["comp-2"] = res_pca_tsne[:, 1]
    df["y"] = cat_res[0]
    df["word"] = list(df_group.index)
    df["label"] = cat_res[1]
    fig = px.scatter(df, x="comp-1", y="comp-2", custom_data=["word", "label"], color=df.y.tolist())
    fig.update_traces(
        hovertemplate="<br>".join([
            "ColX: %{customdata[0]}",
            "Col1: %{customdata[1]}"
        ])
    )
    fig.show()
    # Saving an interactive plot to a plot
    save_html_plots_to_directory(fig, "group")


def map_tsne_one_group_gen(res_pca_tsne, df_group):
    gender_df = df_group["gender"].tolist()
    cat_res = get_categories(df_group)
    df = pd.DataFrame()
    df["comp-1"] = res_pca_tsne[:, 0]
    df["comp-2"] = res_pca_tsne[:, 1]
    df["y"] = cat_res[0]
    df["word"] = list(df_group.index)
    df["label"] = cat_res[1]
    df["gender"] = np.array(gender_df)
    fig = px.scatter(df, x="comp-1", y="comp-2", custom_data=["word", "label", "gender"], color="gender",
                     symbol=df.y.tolist())
    fig.update_traces(
        hovertemplate="<br>".join([
            "ColX: %{customdata[0]}",
            "Col1: %{customdata[1]}"
        ])
    )
    fig.show()
    # Saving an interactive plot to a plot
    save_html_plots_to_directory(fig, "group_gender")


# Color on the basis of the semantic group and the shape on the basis of the gender
def map_tsne_gen(res_pca_tsne, data_frame, categories, category_labels):
    gender_col = data_frame["gender"].tolist()
    df = pd.DataFrame()
    df["comp-1"] = res_pca_tsne[:, 0]
    df["comp-2"] = res_pca_tsne[:, 1]
    df["y"] = categories
    df["word"] = list(data_frame.index)
    df["label"] = category_labels
    df["gender"] = np.array(gender_col)
    fig = px.scatter(df, x="comp-1", y="comp-2", custom_data=["word", "label", "gender"], color=df.y.tolist(),
                     symbol="gender")
    fig.update_traces(
        hovertemplate="<br>".join([
            "ColX: %{customdata[0]}",
            "Col1: %{customdata[1]}"
        ])
    )
    fig.show()
    # Saving an interactive plot to a plot
    save_html_plots_to_directory(fig, "group_gender")


# Color on the basis of gender, no groups
def map_tsne_color_gen(res_pca_tsne, data_frame, categories, category_labels):
    gender_col = data_frame["gender"].tolist()
    df = pd.DataFrame()
    df["comp-1"] = res_pca_tsne[:, 0]
    df["comp-2"] = res_pca_tsne[:, 1]
    df["y"] = categories
    df["word"] = list(data_frame.index)
    df["label"] = category_labels
    df["gender"] = np.array(gender_col)
    fig = px.scatter(df, x="comp-1", y="comp-2", custom_data=["word", "label", "gender"], color=df.gender.tolist())
    fig.update_traces(
        hovertemplate="<br>".join([
            "ColX: %{customdata[0]}",
            "Col1: %{customdata[1]}"
        ])
    )
    fig.show()
    # Saving an interactive plot to a plot
    save_html_plots_to_directory(fig, "group_gender")


""" 
    get_cos_sim function calculates the cosine_similarity between two vectors,
    the parameters are df_column (dataframe with the necessary column name) 
    and sum_av_base (the sum of the base vector and the corresponding average shift vector)
"""
def get_cos_sim(df_column, sum_av_base, ft):
   res_sim = []
   word_vectors = [ft.get_word_vector(x) for x in df_column]
   for i in range(len(word_vectors)):
       similarity = cosine_similarity(word_vectors[i].reshape(1, -1), sum_av_base[i].reshape(1, -1))[0][0]
       res_sim.append(similarity)
   return res_sim


# Calculating weighted average for groups action fem, mas, neut and for the groups food mas, fem, neut
def average_weight(df_mean_fem, df_mean_mas, df_mean_neut):
   # number of lemmas, weight
   num_fem = len(df_mean_mas)
   num_mas = len(df_mean_mas)
   num_neut = len(df_mean_neut)
   # multiplication of each value with its weight
   fem_sum = [num_fem * df_mean_fem[i] for i in range(len(df_mean_fem))]
   mas_sum = [num_mas * df_mean_mas[i] for i in range(len(df_mean_mas))]
   neut_sum = [num_neut * df_mean_neut[i] for i in range(len(df_mean_neut))]
   # sum up the products
   sum_general = [(neut_sum[i] + fem_sum[i] + mas_sum[i]) for i in range(len(list_forms))]
   # divide the sum of products by the sum of weights
   weighted_average = [(sum_general[i] / (num_mas + num_fem + num_neut)) for i in range(len(list_forms))]
   return weighted_average


# Calculating the sum for each case, number to get general mistake for the whole case_number group
def gen_shift_error(case_number):
   result_gen_shift_error = np.sum(np.abs(case_number))
   return result_gen_shift_error


# Different version of perplexity for acc pl
# t-SNE after PCA
def run_pca_tsne_perp(vectors):
    # Calculating PCA and then t-SNE
    pca_50 = PCA(n_components=50)
    tsne = TSNE(n_components=2, perplexity=50)
    """
        fit_transform: fit the model with X and apply the dimensionality reduction on X; returns transformed values
        X(training data): array-like of shape (n_samples, n_features) where n_samples is the number of samples 
        and n_features is the number of features
    """
    pca_result_50 = pca_50.fit_transform(vectors)
    pca_tsne = tsne.fit_transform(pca_result_50)
    return pca_tsne


def save_results_as_csv(data_frame, file_name):
    results_directory = os.path.join("results")
    csv_path = os.path.join(results_directory, f"{file_name}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    data_frame.to_csv(csv_path)

