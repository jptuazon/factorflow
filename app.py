# Copyright 2026 Justin Philip Tuazon

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

# FactorFlow V3.7.1
# https://factorflow-efa.streamlit.app/

import warnings
import base64
import json
import math
import time
from itertools import product, combinations
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from groq import Groq
from factor_model_trainer import InterpretableFA, get_chi_sq, get_df, polychoric
from factor_model_trainer import get_multiset, get_v_index, sign_test
from streamlit_js_eval import streamlit_js_eval
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_extras.card_selector import card_selector
from streamlit_extras.floating_button import floating_button
from streamlit_extras.scroll_to_element import scroll_to_element
from streamlit_lottie import st_lottie
from streamlit_extras.avatar import avatar

# App constants
VERSION_NUMBER = "3.7.1"
ORTHOGONAL_ROTATIONS = ["Priorimax", "Varimax", "Oblimax", "Quartimax", "Equamax"]
OBLIQUE_ROTATIONS = ["Promax", "Oblimin", "Quartimin"]
ROTATIONS = ORTHOGONAL_ROTATIONS + OBLIQUE_ROTATIONS + ["None"]
ESTIMATION_METHODS = {
    "Minimum Residual": "minres",
    "Maximum Likelihood": "ml",
    "Principal Axis Factoring": "principal"
}
COLOR_DISCRETE_SCALES = {
    k: v for k, v in px.colors.qualitative.__dict__.items()
    if isinstance(v, list) and not k.startswith("_")
}
COLOR_CONTINUOUS_DIVERGING_SCALES = {
    k: v for k, v in px.colors.diverging.__dict__.items()
    if isinstance(v, list) and not k.startswith("_")
}
COLOR_CONTINUOUS_SEQUENTIAL_SCALES = {
    k: v for k, v in px.colors.sequential.__dict__.items()
    if isinstance(v, list) and not k.startswith("_")
}
COLOR_SPECIAL_SEQUENTIAL = ["Viridis", "Plasma", "Cividis", "Magma", "Inferno"]


# Universal sentence encoder set up
def load_use_model():
    js_script = f"""
    const load_library = (src) => new Promise((res, rej) => {{
        if (document.querySelector(`script[src="${{src}}"]`)) return res();
        const s = document.createElement('script');
        s.src = src;
        s.onload = res;
        s.onerror = rej;
        document.head.appendChild(s);
    }});

    (async() => {{        
        if (!window.parent.use_model) {{
            await load_library('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');
            await load_library('https://cdn.jsdelivr.net/npm/@tensorflow-models/universal-sentence-encoder');
            window.parent.use_model = await use.load();
        }}

        return 1;
    }})();
    """

    return streamlit_js_eval(js_expressions=js_script, key="load_use_model")


def get_embeddings(sentences):
    js_script = f"""
    (async() => {{
        if (!window.parent.use_model) {{
            return 0;
        }}

        const input_sentences = {json.dumps(sentences)}
        const tensor = await window.parent.use_model.embed(input_sentences);
        const result = await tensor.array();

        tensor.dispose();

        window.parent.use_embeddings = result;

        return result;
    }})();
    """

    embeddings = streamlit_js_eval(js_expressions=js_script, key="get_embeddings")

    return None if embeddings is None else np.array(embeddings)


def clear_embeddings():
    js_script = f"""
    window.parent.use_embeddings = null;
    """

    streamlit_js_eval(js_expressions=js_script, key="clear_embeddings")


st.session_state.USE_MODEL_LOADED = load_use_model()

# Session state
# # Page management
if "CURRENT_PAGE" not in st.session_state:
    st.session_state.CURRENT_PAGE = None

# # State management for fitting factor models
if "SHOW_FIT_DIALOG" not in st.session_state:
    st.session_state.SHOW_FIT_DIALOG = False
if "RUN_FIT" not in st.session_state:
    st.session_state.RUN_FIT = False
if "FIT_DONE" not in st.session_state:
    st.session_state.FIT_DONE = False
if "FIT_SUCCESS" not in st.session_state:
    st.session_state.FIT_SUCCESS = False
if "FIT_ERROR" not in st.session_state:
    st.session_state.FIT_ERROR = ""
if "FIT_MODEL" not in st.session_state:
    st.session_state.FIT_MODEL = "No"

# # State management for calculating polychoric correlations
if "SHOW_POLY_DIALOG" not in st.session_state:
    st.session_state.SHOW_POLY_DIALOG = False
if "RUN_POLY" not in st.session_state:
    st.session_state.RUN_POLY = False
if "POLY_DONE" not in st.session_state:
    st.session_state.POLY_DONE = False
if "POLY_SUCCESS" not in st.session_state:
    st.session_state.POLY_SUCCESS = False
if "POLY_ERROR" not in st.session_state:
    st.session_state.POLY_ERROR = ""
if "CALCULATE_POLY_CORR" not in st.session_state:
    st.session_state.CALCULATE_POLY_CORR = "No"

# # State management for back to top button
if "SCROLL_COUNTER" not in st.session_state:
    st.session_state.SCROLL_COUNTER = 0

# # State management for dataset
if "DATA" not in st.session_state:
    st.session_state.DATA = None
if "DATA_NAME" not in st.session_state:
    st.session_state.DATA_NAME = None
if "IS_LIKERT" not in st.session_state:
    st.session_state.IS_LIKERT = None
if "LIKERT_DIRECTION" not in st.session_state:
    st.session_state.LIKERT_DIRECTION = None
if "STATEMENTS" not in st.session_state:
    st.session_state.STATEMENTS = None
if "STATEMENTS_DF" not in st.session_state:
    st.session_state.STATEMENTS_DF = None
if "VARIABLE_TAGS" not in st.session_state:
    st.session_state.VARIABLE_TAGS = {}

# # State management for NLP
if "EMBEDDINGS" not in st.session_state:
    st.session_state.EMBEDDINGS = None
elif st.session_state.STATEMENTS is not None:
    st.session_state.EMBEDDINGS = get_embeddings(st.session_state.STATEMENTS)
if "SEMANTIC_SIMILARITY_MATRIX" not in st.session_state:
    st.session_state.SEMANTIC_SIMILARITY_MATRIX = None
elif st.session_state.DATA is not None and st.session_state.EMBEDDINGS is not None:
    dots = np.inner(st.session_state.EMBEDDINGS, st.session_state.EMBEDDINGS)
    for i in product(range(dots.shape[0]), range(dots.shape[0])):
        dots[i[0], i[1]] = min(max(-1, dots[i[0], i[1]]), 1)
    semantic_similarity_mat = 1 - (1 / math.pi) * np.arccos(dots)
    semantic_similarity_mat = pd.DataFrame(semantic_similarity_mat,
                                           columns=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])])
    semantic_similarity_mat.index = [f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])]
    st.session_state.SEMANTIC_SIMILARITY_MATRIX = semantic_similarity_mat

# # State management for factor models
if "FACTOR_MODELS" not in st.session_state:
    st.session_state.FACTOR_MODELS = {}
if "FIT_DETAILS" not in st.session_state:
    st.session_state.FIT_DETAILS = {}

# # State management for computed polychoric correlations
if "LARGEST_POLY_CORR" not in st.session_state:
    st.session_state.LARGEST_POLY_CORR = None

# # State management for factor model fit parameters
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "number_of_factors" not in st.session_state:
    st.session_state.number_of_factors = None
if "corr_type" not in st.session_state:
    st.session_state.corr_type = None
if "estimation_method" not in st.session_state:
    st.session_state.estimation_method = None
if "rotation" not in st.session_state:
    st.session_state.rotation = None
if "prior_matrix" not in st.session_state:
    st.session_state.prior_matrix = None
if "prior" not in st.session_state:
    st.session_state.prior = None
if "manifest_vars" not in st.session_state:
    st.session_state.manifest_vars = None

# # State management for demo
if "SAMPLE_V_DATA" not in st.session_state:
    st.session_state.SAMPLE_V_DATA = pd.read_csv("./sample_data/sample_v_data.csv")

# # State management for interpretation
if "CURRENT_LLM_MODEL_ID" not in st.session_state:
    st.session_state.CURRENT_LLM_MODEL_ID = None
if "INTERPRETATIONS" not in st.session_state:
    st.session_state.INTERPRETATIONS = {}


# Delete factor model
def delete_factor_model(model_name):
    if model_name in st.session_state.FACTOR_MODELS:
        del st.session_state.FACTOR_MODELS[model_name]
    if model_name in st.session_state.FIT_DETAILS:
        del st.session_state.FIT_DETAILS[model_name]
    if model_name in st.session_state.INTERPRETATIONS:
        del st.session_state.INTERPRETATIONS[model_name]


# Use cases
@st.dialog(":material/ink_highlighter_move: Use cases", width="medium")
def use_cases_dialog():
    st.subheader(":material/cognition_2: Exploratory Factor Analysis")
    st.markdown("""
    FactorFlow focuses on exploratory factor analysis (EFA), which allows you to explore, identify, and study
    **latent phenomena or constructs**. EFA is used in various fields, including, but not limited to:
    """)
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("""
        * Psychology and Psychometrics (e.g., behavioral traits, scale development)
        * Education (e.g., learning styles)
        * Health and Biology (e.g., characterization of syndromes, dimensionality reduction in genomics)
        * Social Sciences (e.g., social constructs such as social capital)
        """)
    with col_2:
        st.markdown("""
        * Marketing and Consumer Research (e.g., customer perceptions)
        * Organizational Development (e.g., job satisfication, productivity drivers)
        * Economics and Finance (e.g., identification of economic forces)
        * Data Science and Machine Learning (e.g., feature engineering)
        """)

    st.subheader(":material/search_insights: Data Analysis")
    st.markdown("""
    FactorFlow also provides facilities for basic **exploratory data analysis** of the raw dataset. For 
    example, you can quickly see summary statistics (e.g., central tendency, dispersion, skewness, associations) for 
    the variables or features, check distributions, and even **hypothesis testing**. This is useful for a wide class 
    of problems, including **questionnaire or survey analysis**.
    """)

    st.space("xxsmall")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.image("./images/steps/step_6_1.jpeg", width="stretch")
    with col_2:
        st.image("./images/steps/step_8_6.jpeg", width="stretch")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.image("./images/steps/step_8_1.jpeg", width="stretch")
    with col_2:
        st.image("./images/steps/step_8_2.jpeg", width="stretch")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.image("./images/steps/step_4_1.jpeg", width="stretch")
    with col_2:
        st.image("./images/steps/step_10_1.jpeg", width="stretch")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.image("./images/steps/step_8_3.jpeg", width="stretch")
    with col_2:
        st.image("./images/steps/step_8_4.jpeg", width="stretch")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.image("./images/steps/step_5_1.jpeg", width="stretch")
    with col_2:
        st.image("./images/steps/step_7_2.jpeg", width="stretch")


# Specific how-to
@st.dialog(":material/component_exchange: Step-by-step guide", width="large")
def guide_dialog():
    st.markdown("""
    If you're interested only in performing exploratory data analysis on the raw dataset, you can read up to Step 4. 
    If you want to utilize all features (e.g., when performing factor analysis), please read until the end.
    """)
    step = 0

    with st.expander("Pre-requisite: The parts of the app and preferences"):
        st.markdown("""
        Familiarize yourself with the parts of the app. The app has a **Menu** **sidebar**, with four different 
        **panels**:
        * **NLP Models** - This shows the statuses of the NLP models used in the app. You can also configure the large 
        language model here.
        * **Data** - This is where you can upload your datasets. You can also examine your dataset in this panel by 
        clicking the **Stats** button. High-level information is also displayed here.
        * **Factor Models** - You can fit new factor models using this panel. It also shows the factor models that 
        you have fitted and saved so far. You can examine each saved models by clicking **View**. High-level 
        information is also displayed here.
        * **Settings** - You can configure the application to match your preferences. You can change things like the 
        color palettes used, the chart styles, and the font size.

        The app also has five different **pages**:
        * **Overview** - This is the home page, where you can find general information about the app.
        * **Diagnostics** - This is where you can examine diagnostics for factor analysis. Here, you can determine 
        how many factors to use or which manifest variables have low communalities, and so on.
        * **Dashboard** - In this page, you can see multiple figures and visualizations for assessing the factor 
        model. You can also compare two factor models at the same time here.
        * **About** - You can find development details for the app here.
        * **Privacy** - Information about how user uploads are managed can be found here.
        """)

        st.space("xxsmall")

        st.image("./images/steps/step_pre_req_1.png", width="stretch")

        st.space("xxsmall")

        st.markdown("""
        You can also configure app settings (e.g., color palettes) in the **Settings** panel. You can change settings 
        anytime.
        """)

        st.image("./images/steps/step_pre_req_2.png", width="stretch")

    step += 1
    with st.expander(f"Step {step}: Preparing uploads"):
        st.markdown("""
        Prepare your datasets for upload. For a demo, you can also **download sample files** "ECR_data_clean.csv", 
        "ECR_questions.txt", and "sample_statement_tags.csv" from 
        [here](https://drive.google.com/drive/folders/1nc-pZFM5JdxmMrqE_QJyf03DLTEoEH0X) as examples. There are 
        three kinds of datasets or files that you can upload.
        """)

        st.markdown("""
        #### The main dataset (CSV file). 
        This is the tabular dataset on which the factor models will be fit. Each 
        column must represent a feature and each observation must represent an observation. All data values 
        must be numeric and there must have no missing values. This is **required** to fit a model. The CSV file 
        or raw dataset **should not** have column headers. The tool will automatically label the columns as 
        X1, X2, and so on.
        """)

        st.markdown("""
        #### The statements associated with the features (TXT file).
        This is the list of questions or statements associated with each feature in the main dataset. It must 
        be a text file, where statements are separated by linebreaks - consecutive lines with one statement 
        per line. The order of the statements must match the order of the columns in the main dataset (i.e., 
        the first statement must correspond to the first feature). This is an **optional** input, and will be 
        used only if you select "semantics" for the prior in pairwise target rotation.

        Ideally, statements should not be too long but they should also be "complete" (e.g., 
        a full sentence). However, it is also possible to have just "regular" one or two-word variable names 
        such as "height" and so on. In such cases though, semantic similarities may not be as meaningful.
        """)

        st.markdown("""
        #### The tags associated with each variable (manual or CSV file). 
        You can add tags to each variable or statement to help visualize interpretability. 
        To do so, click **Tags** under **Data**. Then, either upload manually or an appropriate CSV file. 
        This is optional.

        The CSV file must have two columns. The first column must contain the variables 
        and the second column must contain the corresponding tags. If a variable has 
        multiple tags, separate them using commas (e.g., "Tag A, Tag B"). If a variable 
        has no tags, leave the cell blank. The CSV file must not have headers.
        """)

    step += 1
    with st.expander(f"Step {step}: Uploading data"):
        st.markdown("""
        Upload "ECR_data_clean.csv" and "ECR_questions.txt" by clicking the **Upload** button in **Data** panel under 
        **Menu**. You should see something like this.
        """)

        st.image("./images/steps/step_2_1.jpeg", width="stretch")
        st.image("./images/steps/step_2_2.jpeg", width="stretch")

        st.markdown("""
        Calculating polychoric correlations can take some time, usually about a minute or two. Wait for it to 
        finish and click "Finish".
        """)

        st.image("./images/steps/step_2_3.jpeg", width="stretch")
        st.image("./images/steps/step_2_4.jpeg", width="stretch")

    step += 1
    with st.expander(f"Step {step}: Tagging variables"):
        st.markdown("""
        This step is optional. However, if you want to upload tags (metadata) for the variables, click **Tags** in 
        the **Data** panel under **Menu**. You can manually tag variable or upload a CSV file. For demonstration 
        purposes, you may upload "sample_statement_tags.csv" for the tags, as shown below.
        """)

        st.image("images/steps/step_3_1.jpeg", width="stretch")
        st.image("images/steps/step_3_2.jpeg", width="stretch")

    step += 1
    with st.expander(f"Step {step}: Summary statistics and hypothesis testing"):
        st.markdown("""
        You can view **summary statistics** and conduct **hypothesis testing** in the **View basic stats** window. To 
        get to it, click **Stats** in the **Data** panel under **Menu**.
        """)

        st.image("./images/steps/step_4_1.jpeg", width="stretch")
        st.image("./images/steps/step_4_2.jpeg", width="stretch")

    step += 1
    with st.expander(f"Step {step}: Diagnostics"):
        st.markdown("""
        You can check out the diagnostics for factor analysis in the **Diagnostics** tab. This will help you decide 
        which manifest variables to retain and how many factors to include. You can find the following:
        * **Goodness-of-fit** - Various goodness-of-fit metrics are displayed here.
        * **Communalities and adequacies** - You can see the communality and Kaiser-Meyer-Olikin sampling adequacy for 
        each manifest variable here.
        * **Scree plot** - This shows the sum of squared loadings (i.e., eigenvalue) per additional factor included.
        * **Correlations** - This shows the pairwise correlations for the selected manifest variables.

        **Goodness-of-fit** and **Correlations** should help you assess the global or overall quality of fit of a 
        factor model. On the other hand, **Communalities and adequacies** can help you identify which variables to 
        keep while **Scree plot** can help you detmrine the number of factors to include.
        """)

        st.image("./images/steps/step_5_1.jpeg", width="stretch")

    step += 1
    with st.expander(f"Step {step}: Fitting a factor model"):
        st.markdown("""
        Fit at least one factor model. You can input the model's name, the number of factors, the type of correlation 
        to use, the manifest variables included in the model, the estimation method, the rotation, and the prior 
        matrix.
        """)

        st.image("./images/steps/step_6_1.jpeg", width="stretch")

    step += 1
    with st.expander(f"Step {step}: Viewing factor models"):
        st.markdown("""
        In the **Factor Models** panel under **Menu**, click **View** to see your saved models. Here, you can check each 
        model's fit details, correlation matrix, prior matrix (if present), loadings, communalities, factor scores, 
        factor correlations, and more. You can also delete models here.
        """)

        st.image("./images/steps/step_7_1.jpeg", width="stretch")
        st.image("./images/steps/step_7_2.jpeg", width="stretch")

    step += 1
    with st.expander(f"Step {step}: Using the dashboard"):
        st.markdown("""
        Head on over to the **Dashboard** tab, which is where most of the information and visualizations that you need 
        for assessing a factor model can be found. Select the model that you want to examine. There are several 
        sections here:
        * **Fit details** - This shows information about the model's fit details (e.g., rotation, correlation type).
        * **Communalities and adequacies** - This shows the communality and adequacy for each manifest variable included 
        in the factor model.
        * **Interpretability plot** - This shows the scatterplot, with a locally weighted scatterplot 
        smoothing (LOWESS) curve, on semantic (or prior) similarity vs loading similarity. This visualization can 
        tell you how much the loading similarities agree with the semantic (or prior) similarities. The V-index is 
        displayed below the title of the plot.
        * **Factor breakdown** - This shows how much of each tag each factor is composed of, in terms of sum of squared 
        loadings. This can help you interpret the factors based on the tags that you have provided (if any). For 
        example, if Factor 1's sum of squared loadings comes mostly from Tag A, then you can associate the meaning of 
        Factor 1 with Tag A.
        * **Factor loadings** - This shows you the empirical distribution (in terms of the cumulatie distribution 
        function) of the absolute loadings, both for each factor and for the whole model. You can use the distribution 
        plots to asses which threshold for the absolute loadings make sense. Below the distribution plot, the factor 
        loading matrix, presented as a heatmap, can be seen. You can choose whether to show the exact loadings or the 
        binarized loadings. This will help you define what each factor is.
        * **Loadings comparison** - The plot here allows you to compare the loadings across manifest variables or 
        across factors. For example, when comparing across factors, if two lines are coincident or parallel, then 
        the corresponding factors are similar (in terms of definition).
        * **Factor cross-loadings** - This network graph tells you how much the factors overlap. For example, if the 
        "network" of Factor 1 shares multiple nodes with the "network" of Factor 2, there are multiple cross-loadings 
        (i.e., significant overlap) between the two factors.
        * **Interpretation** - You can generate automatic interpretations for factor models (at a factor-level) here 
        using the selected large language model.
        """)

        st.space("xxsmall")

        col_1, col_2 = st.columns(2)
        with col_1:
            st.image("./images/steps/step_8_1.jpeg", width="stretch")
        with col_2:
            st.image("./images/steps/step_8_2.jpeg", width="stretch")

        col_1, col_2 = st.columns(2)
        with col_1:
            st.image("./images/steps/step_8_3.jpeg", width="stretch")
        with col_2:
            st.image("./images/steps/step_8_4.jpeg", width="stretch")

        col_1, col_2 = st.columns(2)
        with col_1:
            st.image("./images/steps/step_8_5.jpeg", width="stretch")
        with col_2:
            st.image("./images/steps/step_8_6.jpeg", width="stretch")

    step += 1
    with st.expander(f"Step {step}: Generating interpretations"):
        st.markdown("""
        In the **Interpretation** section of the **Dashboard** tab, you can easily generate an automated interpretation 
        for the factor model using the large language model selected (shown in the **NLP Models** panel under **Menu**). 
        You can choose which model to use and with which temperature to generate, and even re-generate interpretations 
        multiple times.
        """)

        st.image("./images/steps/step_9_1.jpeg", width="stretch")

    step += 1
    with st.expander(f"Step {step}: Comparing models"):
        st.markdown("""
        You can select up to two factor models simultaneously in **Dashboard**, which allows you to easily compare 
        and contrast factor models, and ultimately select the final factor model. The first model will be shown 
        on the left while the second one will be shown on the right.
        """)

        st.image("./images/steps/step_10_1.jpeg", width="stretch")

    step += 1
    with st.expander(f"Step: {step} Downloading results"):
        st.markdown("""
        Almost all tables and visualizations can be downloaded from the app. Some have dedicated download buttons 
        as shown below.
        """)

        st.image("./images/steps/step_11_1.jpeg", width="stretch")

        st.markdown("""
        For most visualizations, you can hover at the top right of the chart and click the camera icon. This will 
        download the chart as a PNG file. For tables, you can also hover at the top right and click the download icon 
        to download them as CSV files. 
        """)

        st.image("./images/steps/step_11_2.png", width="stretch")
        st.image("./images/steps/step_11_3.png", width="stretch")

        st.markdown("""
        Finally, for a few visualizations, you can download them by right-cliking and clicking **Save image as**.
        """)

        st.image("./images/steps/step_11_4.png", width="stretch")


# Fit factor model
if st.session_state.FIT_MODEL == "Yes":
    st.session_state.SHOW_FIT_DIALOG = True
    st.session_state.RUN_FIT = True
    st.session_state.FIT_MODEL = "No"


@st.dialog("Fit a new model", dismissible=False)
def fit_factor_model():
    if st.session_state.RUN_FIT:
        st.session_state.RUN_FIT = False

        suceeded = True
        error_msg = ""

        with st.spinner("Fitting the factor model...", width="stretch", show_time=True):
            try:
                model_name = str(st.session_state.model_name)
                number_of_factors = int(st.session_state.number_of_factors)
                corr_type = st.session_state.corr_type
                estimation_method_key = st.session_state.estimation_method
                estimation_method = ESTIMATION_METHODS[estimation_method_key]
                rotation = None if st.session_state.rotation == "None" else str(st.session_state.rotation).lower()
                prior_matrix = st.session_state.prior_matrix
                manifest_vars = st.session_state.manifest_vars

                if rotation == "equamax":
                    rot_kwargs = {
                        "kappa": number_of_factors / (2 * len(manifest_vars))
                    }
                else:
                    rot_kwargs = None

                if corr_type == "Pearson":
                    st.session_state.FACTOR_MODELS[model_name] = InterpretableFA(
                        st.session_state.DATA[manifest_vars]
                    )
                elif corr_type == "Polychoric":
                    df_poly_corr_mat = st.session_state.LARGEST_POLY_CORR.loc[manifest_vars, manifest_vars]
                    st.session_state.FACTOR_MODELS[model_name] = InterpretableFA(
                        df_poly_corr_mat, True, st.session_state.DATA.shape[0]
                    )

                st.session_state.FACTOR_MODELS[model_name].fit_factor_model(
                    model_name=model_name,
                    n_factors=number_of_factors,
                    method=estimation_method,
                    rotation=rotation,
                    prior=prior_matrix,
                    rotation_kwargs=rot_kwargs
                )

                st.session_state.FIT_DETAILS[model_name] = {
                    "number_of_factors": number_of_factors,
                    "corr_type": corr_type,
                    "estimation_method": estimation_method_key,
                    "rotation": rotation,
                    "prior": st.session_state.prior,
                    "manifest_vars": manifest_vars
                }

                st.session_state.INTERPRETATIONS[model_name] = (None, "")
            except Exception as e:
                delete_factor_model(model_name)
                suceeded = False
                error_msg = str(e)

        st.session_state.FIT_DONE = True
        st.session_state.FIT_SUCCESS = suceeded
        st.session_state.FIT_ERROR = error_msg

        st.session_state.model_name = None
        st.session_state.number_of_factors = None
        st.session_state.corr_type = None
        st.session_state.estimation_method = None
        st.session_state.rotation = None
        st.session_state.prior_matrix = None

        if st.session_state.FIT_DONE:
            if st.session_state.FIT_SUCCESS:
                st.success("Successfully fit factor model.")
            else:
                st.error(f"""
                Failed to fit factor model.

                Error message: {st.session_state.FIT_ERROR}
                """)

    st.space()
    _, col_2 = st.columns([15, 3])
    with col_2:
        if st.button("Finish", width="stretch", key="finish_fit_btn"):
            st.session_state.SHOW_FIT_DIALOG = False
            st.session_state.FIT_DONE = False
            st.rerun()


if st.session_state.SHOW_FIT_DIALOG:
    fit_factor_model()

# Calculate polychoric
if st.session_state.CALCULATE_POLY_CORR == "Yes":
    st.session_state.SHOW_POLY_DIALOG = True
    st.session_state.CALCULATE_POLY_CORR = "No"


@st.dialog(":material/calculate: Calculating polychoric correlations", dismissible=False)
def calculate_poly_corr():
    if not st.session_state.get("POLY_DONE", False):
        suceeded = True
        error_msg = ""

        try:
            prog_txt = "Computing polychoric correlations..."
            prog_bar = st.progress(0, text=prog_txt)

            df_data = st.session_state.DATA
            data_arr = df_data.to_numpy(dtype=int)

            var_count = data_arr.shape[1]
            poly_corr_mat = np.eye(var_count)
            current_iter = 0
            total_iter = (var_count * (var_count - 1)) // 2

            for row in range(var_count):
                for col in range(row):
                    corr = polychoric(
                        data_arr[:, row],
                        data_arr[:, col]
                    )
                    poly_corr_mat[row, col] = corr
                    poly_corr_mat[col, row] = corr

                    current_iter += 1
                    if current_iter == total_iter:
                        prog_txt = "Done."
                    prog_bar.progress(current_iter / total_iter, text=prog_txt)

            st.session_state.LARGEST_POLY_CORR = pd.DataFrame(
                poly_corr_mat,
                columns=df_data.columns,
                index=df_data.columns
            )
        except Exception as e:
            suceeded = False
            error_msg = str(e)

        st.session_state.POLY_DONE = True
        st.session_state.POLY_SUCCESS = suceeded
        st.session_state.POLY_ERROR = error_msg

    if st.session_state.POLY_DONE:
        if st.session_state.POLY_SUCCESS:
            st.success("Successfully computed polychoric correlations.")
        else:
            st.error(f"""
            Failed to compute polychoric correlations.

            Error message: {st.session_state.POLY_ERROR}
            """)

    st.space()
    _, col_2 = st.columns([15, 3])
    with col_2:
        if st.button("Finish", width="stretch", key="finish_poly_btn"):
            st.session_state.SHOW_POLY_DIALOG = False
            st.session_state.POLY_DONE = False
            st.rerun()


if st.session_state.SHOW_POLY_DIALOG:
    calculate_poly_corr()

# LLM set up
LLM_API_KEY = st.secrets["GROQ_API_KEY"]
LLM_MODEL_IDS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

if "LLM_CLIENT" not in st.session_state:
    st.session_state.LLM_CLIENT = Groq(api_key=LLM_API_KEY)


def generate_interpretation(factors):
    llm_model_id = st.session_state.CURRENT_LLM_MODEL_ID
    try:
        interpretation = st.session_state.LLM_CLIENT.chat.completions.create(
            model=llm_model_id,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert in Exploratory Factor Analysis (EFA). Your role is to act as an 
                    "EFA Factor Interpretation Assistant".

                    For each factor, you must:
                    1. Rewrite and enumerate the statements as "Statement 1", "Statement 2", and so on.
                    2. Generate a concise factor label (1 to 4 words only).
                    3. Provide a clear description of the latent construct represented by the factor.
                    4. Provide a justification explaining your interpretations.

                    Important instructions to follow (EXTREMELY STRICT):
                    - Each factor MUST include ALL four sections: Statements, Label, Description, and Justificiation.
                    - Process ALL factors. Do NOT omit any section for any factor.
                    - All factors must follow the exact same output structure and format.
                    - Do NOT quote full statements outside the "Statements" section.
                    - Each factor MUST include ALL four sections: Statements, Label, Description, and Justification.
                    - Do NOT omit any section for any factor. Always refer to statements using their labels 
                      (e.g., "Statement 1").
                    - Adhere to all rules and requirements listed next.

                    Statements Section Requirements:
                    - List ALL statements under the factor.
                    - The statement label is given before each statement. The loading's sign is also given. For 
                    example, in "X1 (Positive): I am sad.", X1 is the statement label of "I am sad." and the sign of 
                    the loading is positive.
                    - Format as:
                      [statement label 1] (loading sign 1): [full statement 1]
                      [statement label 2] (loading sign 2): [full statement 2]
                    - Preserve the original wording exactly (do NOT paraphrase).
                    - Label the statements with the labels given in the input and list them in the order that
                    they are given.
                    - Within a factor, list down each statement included ONLY once. This is important.

                    Description Requirements:
                    - Avoid surface-level or generic interpretations.
                    - Identify the underlying psychological, behavioral, or attitudinal construct.
                    - Prefer abstract constructs over literal summaries of statements.
                    - Take into account the signs of the loadings when creating an interpretation.

                    Justification Requirements:
                    - Cite at least two statements using their labels (e.g., "X1").
                    - Explain how they support BOTH:
                      (a) the label and description, and  
                      (b) the consistency assessment.
                    - Go beyond restating. Provide reasoning.
                    - Take into account the signs of the loadings when creating an interpretation.

                    Scale Direction Rule:
                    - At the beginning of the input, it is possible that "Scale Direction" is specified. It can be 
                    either "Disagree-Agree", which means that larger variable values indicate higher agreement levels, 
                    or "Agree-Disagree", which means larger variable values indicate lower agreement levels.
                    - When interpreting factors, take the scale direction into account, if scale direction is 
                    present. 
                      (a) If the loading sign is positive and the scale direction is "Disagree-Agree", greater 
                      agreement with the variable statement is associated with higher factor scores.
                      (b) If the loading sign is positive and the scale direction is "Agree-Disagree", greater 
                      disagreement with the variable statement is associated with higher factor scores.
                      (c) If the loading sign is negative and the scale direction is "Disagree-Agree", greater 
                      agreement with the variable statement is associated with lower factor scores.
                      (d) If the loading sign is negative and the scale direction is "Agree-Disagree", greater 
                      disagreement with the variable statement is associated with lower factor scores.
                    - Note that the scale direction by itself does NOT influence interpretation. However, pairing 
                    the sign of the loading with the scale direction allows you to properly interpret the 
                    "association" of a manifest variable with a factor.
                    - The application of the scale direction rule is on a per-statement or per-variable basis. 
                    After such, the overall interpretation is determined.

                    Conflict Handling Rule:
                    - If statements reflect multiple distinct or conflicting themes:
                      - Identify the dominant theme.
                      - Note secondary or conflicting themes in the Description section.
                      - Do NOT force an artificial single interpretation.

                    Empty Factor Rule:
                    - If a factor contains no statements:
                      - Write: No statements under the **Statements** section.
                      - For Label, Description, and Justification, write: Not applicable.
                      - Do NOT attempt to infer or generate content.

                    Ambiguity Handling Rule:
                    - If a statement is vague or ambiguous:
                      - Acknowledge this in the Justification section.
                      - Explain how this affects interpretation.

                    Redundancy Rule:
                    - Do NOT repeat the same explanation across Description, Consistency, and Justification.
                    - Each section must contribute distinct information.

                    Formatting Rules:
                    - Bold the factor name (e.g., **factor_1**).
                    - Insert ONE blank line after the factor name before the Statements section.
                    - Use bullet points (•) for each section.
                    - Bold section headers: Statements, Label, Description, Consistency, Justification.
                    - Insert one blank line between sections.
                    - Add a separator line between factors: -----------------------------------

                    Output Requirements (EXTREMELY STRICT):
                    - You MUST process ALL factors. Again, ALL factors. Ensure that.
                    - Every factor MUST contain ALL four sections. Again, ALL sections. Ensure that.
                    - Do NOT add or remove sections.
                    - Do NOT add any introductory or concluding text.
                    - Follow formatting EXACTLY.
                    - Follow ALL RULES AND REQUIREMENTS EXACTLY.

                    Self-Check (DO NOT OUTPUT THIS SECTION):
                    Before finalizing your response, internally verify that:
                    - Every factor includes ALL four sections.
                    - No section is missing or incorrectly formatted.
                    - No relevant statement is missing for ANY factor.
                    - "No statements" and "Not applicable" are used correctly when required.
                    - No full statements appear outside the Statements section.
                    - All references to statements use their labels.
                    - Formatting exactly matches the template.
                    - All rules and requirements are followed.
                    - Within a factor, list down each statement included ONLY once. This is important. For instance, 
                    if "X1: I am sad." is included in factor_1, then "X1: I am sad." must appear in factor_1 EXACTLY 
                    once. No duplicates.
                    - Take into account the signs of the loadings when creating an interpretation.

                    Input Template:
                    Scale Direction - [direction of scale]

                    **factor_X**
                    - [statement label 1] [loading sign 1]: [full statement 1]
                    - [statement label 2] (loading sign 2): [full statement 2]

                    Output Template (APPLY TO EVERY FACTOR WITHOUT EXCEPTION):

                    **factor_X**

                    • **Statements**:
                      [statement label 1] (loading sign 1): [full statement 1] 
                      [statement label 2 (loading sign 2): [full statement 2]

                    • **Label**: [2–4 word label]

                    • **Description**: [Explanation of the latent construct]

                    • **Justification**: [Use Statement numbers and explain reasoning]

                    -----------------------------------   
                    """
                },
                {
                    "role": "user",
                    "content": factors
                }
            ],
            max_completion_tokens=4096,
            temperature=llm_temp,
            top_p=0.95,
            stream=False
        )

        finish_reason = interpretation.choices[0].finish_reason

        if finish_reason == "length":
            st.toast("⚠️ LLM output was truncated due to token limits.")
        elif finish_reason == "content_filter":
            st.toast("⚠️ LLM output was omitted due to safety filters.")
        elif finish_reason == "stop":
            st.toast("✅ LLM output was successfully generated.")

        return llm_model_id, interpretation.choices[0].message.content.replace("\n", "  \n")
    except Exception as e:
        st.toast(e)
        st.toast("❌ Cannot use current LLM. Please try again in a while or use a different model.")
        return "Error", "Rate limit reached. Please try again in a while or use a different model."


def interpret_factor_model(df_discretized_loadings, model_name):
    input_for_llm = ""

    if st.session_state.IS_LIKERT:
        direction = "Disagree-Agree" if st.session_state.LIKERT_DIRECTION == "01" else "Agree-Disagree"
        input_for_llm += f"Scale Direction - {direction}\n"

    if df_discretized_loadings is None:
        st.session_state.INTERPRETATIONS[model_name] = (None, "")
        return

    for idx, factor in enumerate(df_discretized_loadings.columns):
        factor_loadings = df_discretized_loadings[factor]
        variables = df_discretized_loadings[factor_loadings.abs() == 1].index.tolist()
        variables = sorted(variables, key=lambda x: int(x[1:]))
        signs = ["Positive" if factor_loadings.loc[variable] > 0 else "Negative" for variable in variables]

        input_for_llm += f"\n{factor}:\n"
        if len(variables) == 0:
            input_for_llm += "- No statements."
            continue
        else:
            for idy, variable in enumerate(variables):
                statement = st.session_state.STATEMENTS_DF[
                    st.session_state.STATEMENTS_DF["Variable"] == variable
                    ]["Statement"].item()
                input_for_llm += f"- {variable} ({signs[idy]}): {statement}\n"

    input_for_llm = input_for_llm.encode("utf-8").decode("unicode_escape")
    interpretation = generate_interpretation(input_for_llm)
    st.session_state.INTERPRETATIONS[model_name] = interpretation


# Page configuration
st.set_page_config(
    page_title="FactorFlow",
    page_icon="./images/ff_icon.png",
    layout="wide",
    initial_sidebar_state="auto"
)


# App functions
def load_lottie_file(path):
    with open(path, "r") as f:
        return json.load(f)


def get_mean_color(hex_colors):
    if not hex_colors:
        return "#000000"

    rgbs = []
    for h in hex_colors:
        h = h.lstrip("#")
        rgbs.append(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))

    total_r = sum(r for r, g, b in rgbs)
    total_g = sum(g for r, g, b in rgbs)
    total_b = sum(b for r, g, b in rgbs)

    num_colors = len(hex_colors)
    avg_r = int(total_r / num_colors)
    avg_g = int(total_g / num_colors)
    avg_b = int(total_b / num_colors)

    return "#{:02x}{:02x}{:02x}".format(avg_r, avg_g, avg_b)


def draw_colored_square(label, color_hex):
    square_html = f"""
    <span style="
        display: inline-block; 
        width: 15px; 
        height: 15px; 
        background-color: {color_hex}; 
        border: 1px solid black; 
        margin-right: 10px; 
        vertical-align: middle;">
    </span>
    """

    st.markdown(f"{square_html}<span style='vertical-align: middle;'>{label}</span>",
                unsafe_allow_html=True)


def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"


def rgb_to_hex(rgb_color):
    rgb_color = rgb_color.strip()
    rgb_color = rgb_color.replace("rgb(", "").replace(")", "")
    r, g, b = map(int, rgb_color.split(","))

    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def compute_tags_breakdown(df_loadings):
    df_loadings = df_loadings.copy().reset_index(drop=False)
    factor_cols = [col for col in df_loadings.columns if col.startswith("factor_")]
    df_loadings[factor_cols] = df_loadings[factor_cols] ** 2

    tag_variable = {}
    for variable, tags in st.session_state.VARIABLE_TAGS.items():
        for tag in tags:
            if tag not in tag_variable:
                tag_variable[tag] = [variable]
            else:
                tag_variable[tag].append(variable)

    tags = []
    factors = []
    squared_loadings_sum = []
    for factor, tag in product(factor_cols, list(tag_variable.keys())):
        variables = tag_variable[tag]
        factor_squared_loadings = df_loadings[["variable", factor]].copy()
        factor_squared_loadings["included"] = factor_squared_loadings["variable"].isin(variables).astype(int)
        total = factor_squared_loadings[
            factor_squared_loadings["included"] == 1
            ][factor].sum()
        tags.append(tag)
        factors.append(factor)
        squared_loadings_sum.append(total)

        del factor_squared_loadings

    tagged_variables = set([variable for variables in tag_variable.values() for variable in variables])
    untagged_variables = [f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])
                          if f"X{idx + 1}" not in tagged_variables]
    for factor in factor_cols:
        variables = untagged_variables
        factor_squared_loadings = df_loadings[["variable", factor]].copy()
        factor_squared_loadings["included"] = factor_squared_loadings["variable"].isin(variables).astype(int)
        total = factor_squared_loadings[
            factor_squared_loadings["included"] == 1
            ][factor].sum()
        tags.append("No tag")
        factors.append(factor)
        squared_loadings_sum.append(total)

        del factor_squared_loadings

    df_tags_breakdown = pd.DataFrame({
        "Factor": factors,
        "Tag": tags,
        "Sum of Squared Loadings": squared_loadings_sum
    })

    del df_loadings

    return df_tags_breakdown


def process_prior_matrix(prior_matrix, rotation, manifest_vars):
    result = {
        "pass": True,
        "message": "Passed.",
        "processed_matrix": None
    }

    prior_matrix = prior_matrix.copy() if prior_matrix is not None else None

    if prior_matrix is None:
        if rotation.lower == "priorimax":
            result["pass"] = False
            result["message"] = "Priormax rotation requires a prior matrix."
        return result

    if len(prior_matrix.shape) != 2:
        result["pass"] = False
        result["message"] = "The custom prior matrix must be 2D."
        return result

    if (prior_matrix.shape[0] != len(manifest_vars)
            or prior_matrix.shape[1] != len(manifest_vars)):
        result["pass"] = False
        result["message"] = ("The number of rows (or columns) of the prior matrix must match the number of manifest "
                             "variables.")
        return result

    for row in range(prior_matrix.shape[0]):
        for col in range(row + 1):
            val = prior_matrix[row, col]

            if val == "":
                val = None
                prior_matrix[row, col] = val
            else:
                try:
                    val = float(val)
                    prior_matrix[row, col] = val
                except ValueError:
                    result["pass"] = False
                    result["message"] = "Prior matrix: All entries must be either a number or blank."
                    return result

                if val < 0 or val > 1:
                    result["pass"] = False
                    result["message"] = "Prior matrix: All entries must be between 0 and 1 (inclusive)."
                    return result

            if val is None:
                if prior_matrix[col, row] is None or prior_matrix[col, row] == "":
                    prior_matrix[col, row] = None
                    continue
                else:
                    result["pass"] = False
                    result["message"] = "The prior matrix must be symmetric."
                    return result
            else:
                if prior_matrix[col, row] is None or prior_matrix[col, row] == "":
                    result["pass"] = False
                    result["message"] = "The prior matrix must be symmetric."
                    return result
                else:
                    if not np.isclose(val, prior_matrix[col, row]):
                        result["pass"] = False
                        result["message"] = "The prior matrix must be symmetric."
                        return result
                    else:
                        prior_matrix[col, row] = val

    result["processed_matrix"] = prior_matrix

    return result


@st.dialog(":material/upload: Upload dataset", width="medium")
def upload_data_dialog():
    st.info("""
    :material/info: Ensure that you have read **Getting started** in the **Overview** page before proceeding.
    """)
    df_data = None
    data_file_name = None
    is_likert = None
    likert_direction = None
    statements = None
    statements_file_name = None
    use_statements = None
    calc_poly_corr = None

    can_proceed = True

    st.space("xxsmall")
    is_likert = st.selectbox(
        "Is your data composed of Likert-type items?",
        options=["Yes", "No"],
        index=0
    )
    if is_likert == "Yes":
        col_1, col_2 = st.columns(2)
        with col_1:
            likert_direction = st.radio(
                "Direction of the scale?",
                options=["Disagree-Agree", "Agree-Disagree"],
                horizontal=True,
                help="""
                  "Disagree-Agree" means that larger numbers indicate stronger levels of agreement, such as 1 - 
                  Strongly Disagree to 5 - Strongly Agree. "Agree-Disagree" refers to the opposite direction, 
                  such as 1 - Very Satisfied to 5 - Very Dissatisfied.
                  """
            )
        with col_2:
            calc_poly_corr = st.radio(
                "Calculate polychoric correlations?",
                options=["Yes", "No"],
                horizontal=True,
                help="""
                Calculating polychoric correlations takes additional time but will enable you to fit factor 
                models with polychoric correlations, which are generally preferred.
                """
            )

    if calc_poly_corr == "Yes":
        st.warning(":material/warning: Note that calculating polychoric correlations may take up to a few minutes.")

    st.divider()

    csv_data = st.file_uploader("Choose a CSV file for the main dataset:", type="csv")
    if csv_data is not None:
        df_data = pd.read_csv(csv_data, header=None)
        df_data.columns = [F"X{i + 1}" for i in range(len(df_data.columns))]
        df_data = df_data.apply(pd.to_numeric, errors="coerce")
        data_file_name = csv_data.name
        if df_data.isnull().values.any() > 0:
            st.error("The dataset must have only numeric values and there must be no missing values.")
            can_proceed = False
        else:
            st.success("This dataset is valid.")

    if can_proceed:
        if df_data is not None:
            st.divider()
            use_statements = st.checkbox("Upload statements or questions associated with the manifest variables?",
                                         key="upload_questions_checkbox")
            if use_statements:
                txt_data = st.file_uploader("Choose a TXT file for the statements:", type="txt")
                if txt_data is not None:
                    statements = txt_data.getvalue().decode("utf-8")
                    statements = statements.splitlines()
                    statements_file_name = txt_data.name
                    if len(statements) != df_data.shape[1]:
                        st.error("The number of statements must match the number of columns in the main dataset.")
                        can_proceed = False
                    elif len(statements) > len(set(statements)):
                        st.error("The statements must be unique.")
                        can_proceed = False
                    else:
                        st.success("This list of statements is valid.")
                else:
                    st.warning("Please upload the CSV file.")
                    can_proceed = False

    st.space()
    if can_proceed and (df_data is not None):
        _, col_2 = st.columns([15, 3])
        with col_2:
            if st.button("Confirm", width="stretch", type="primary", key="confirm_data_btn"):
                st.session_state.DATA = df_data
                st.session_state.DATA_NAME = data_file_name
                st.session_state.IS_LIKERT = (is_likert == "Yes") if is_likert is not None else None
                st.session_state.LIKERT_DIRECTION = (
                    ("01" if likert_direction == "Disagree-Agree" else "10")
                    if likert_direction is not None else None
                )
                st.session_state.STATEMENTS = statements
                st.session_state.STATEMENTS_NAME = statements_file_name
                st.session_state.STATEMENTS_DF = pd.DataFrame({
                    "Variable": [f"X{idx + 1}" for idx in range(df_data.shape[1])],
                    "Statement": statements
                })
                st.session_state.VARIABLE_TAGS = {}
                for idx in range(df_data.shape[1]):
                    st.session_state.VARIABLE_TAGS[f"X{idx + 1}"] = []

                st.session_state.FACTOR_MODELS = {}
                st.session_state.FIT_DETAILS = {}
                st.session_state.INTERPRETATIONS = {}
                st.session_state.LARGEST_POLY_CORR = None

                st.session_state.CALCULATE_POLY_CORR = calc_poly_corr

                st.rerun()


@st.dialog(":material/shoppingmode: Manifest variable tags", width="large")
def view_tags_dialog():
    st.write("""
    Tags are a priori labellings of variables or statements. For instance, you can tag the statements "I am loyal to 
    brands I have used before." and "I associate products with memories." with "Nostalgia". Each variable can have 
    zero or more tags. These tags are then used to summarize the "breakdown" of a factor in the **Dashboard**.
    """)

    df_tags = pd.DataFrame([
        {"Variable": variable, "Tags": ", ".join(tags)}
        for variable, tags in st.session_state.VARIABLE_TAGS.items()
    ])
    df_tags.index = [f"X{int(idx + 1)}" for idx in df_tags.index]
    df_tags = pd.merge(left=df_tags, right=st.session_state.STATEMENTS_DF, on="Variable", how="left")
    df_tags = df_tags[["Variable", "Statement", "Tags"]]
    df_tags.sort_values(by=["Statement", "Variable"], ascending=[True, True], inplace=True)
    st.dataframe(df_tags, key="df_tags")

    can_proceed = True

    st.space()
    _, col_2 = st.columns([6, 1])
    with col_2:
        if st.button("Clear all tags", width="stretch", key="clear_tags_btn"):
            st.session_state.VARIABLE_TAGS = {}
            st.rerun()

    st.markdown("## Edit tags")

    input_type = st.selectbox(
        "Input type",
        options=["Manual", "Upload"],
        key="input_type_tags"
    )
    current_variable = None
    df_new_tags = None

    if input_type == "Manual":
        st.markdown("### Current variable:")

        if st.session_state.STATEMENTS is not None:
            choices = [f"X{i + 1} - {st.session_state.STATEMENTS[i]}"
                       for i in range(st.session_state.DATA.shape[1])]
        else:
            choices = [f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])]

        current_variable = st.selectbox(
            "Choose variable or statement",
            options=choices,
            key="choose_statement_tags"
        )
        current_variable = current_variable.split(" - ")[0]

        current_tags = ", ".join(list(st.session_state.VARIABLE_TAGS[current_variable]))
        st.markdown("### Current tags:")
        if current_tags != "":
            st.write(current_tags)
        else:
            st.write("No tags yet")

        st.markdown("### New tags:")
        new_tags = st.multiselect(
            "Enter new tags",
            options=list(st.session_state.VARIABLE_TAGS[current_variable]),
            accept_new_options=True,
            placeholder="Selecting no tag will remove all tags..",
            key="tags_multi"
        )
    else:
        new_tags = st.file_uploader(label="Upload CSV file for the manifest variable tags", type="csv", width="stretch",
                                    help="""
                                    The CSV file must have two columns. The first column must contain the variables 
                                    and the second column must contain the corresponding tags. If a variable has 
                                    multiple tags, separate them using commas (e.g., "Tag A, Tag B"). If a variable 
                                    has no tags, leave the cell blank. The CSV file must not have headers.
                                    """)
        if new_tags is not None:
            df_new_tags = pd.read_csv(new_tags, header=None)

            if df_new_tags.shape[1] != 2:
                st.error("The CSV file must have two columns, first for the variable and second for the tag.")
                can_proceed = False
            df_new_tags.columns = ["variable", "tags"]

            if set(df_new_tags["variable"]) != set([f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])]):
                st.error("The set of variables in the CSV file must match exactly the set of variables "
                         "originally uploaded.")
                can_proceed = False

    if can_proceed:
        st.space()
        _, col_2 = st.columns([6, 1])
        with col_2:
            if st.button("Confirm", width="stretch", key="confirm_tags_btn"):
                if input_type == "Manual":
                    st.session_state.VARIABLE_TAGS[current_variable] = new_tags
                else:
                    st.dataframe(df_new_tags, key="df_new_tags")
                    for idx, row in df_new_tags.iterrows():
                        variable = row["variable"]
                        tags = row["tags"].split(",") if not pd.isna(row["tags"]) else []
                        tags = [tag.strip() for tag in tags if tag.strip() != ""] if len(tags) > 0 else tags
                        st.session_state.VARIABLE_TAGS[variable] = tags
                st.rerun()


@st.dialog(":material/analytics: View basic stats", width="large")
def view_data_dialog():
    st.info("""
    :material/info: You can download almost all tables (as CSV) and charts (as CSV or PNG) in this app, 
    including the ones below.
    """)

    col_1, col_2 = st.columns(2)
    with col_1:
        st.download_button("Download raw data as CSV file", st.session_state.DATA.to_csv(),
                           file_name="raw_data.csv", width="stretch", type="primary")
    with col_2:
        st.download_button("Download associated statements", st.session_state.STATEMENTS_DF.to_csv(),
                           file_name="associated_statements.csv", width="stretch")

    st.space("xxsmall")

    st.subheader("Summary statistics")
    st.markdown(f"Sample size = **{st.session_state.DATA.shape[0]:,}**")
    df_summary = pd.DataFrame(
        {
            "Variable": [f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])],
            "Statement": st.session_state.STATEMENTS,
            "Mean": [
                st.session_state.DATA[col].mean() for col in st.session_state.DATA.columns
            ],
            "SD": [
                st.session_state.DATA[col].std() for col in st.session_state.DATA.columns
            ],
            "Median": [
                st.session_state.DATA[col].median() for col in st.session_state.DATA.columns
            ],
            "IQR": [
                st.session_state.DATA[col].quantile(0.75) - st.session_state.DATA[col].quantile(0.25)
                for col in st.session_state.DATA.columns
            ],
            "Min": [
                st.session_state.DATA[col].min() for col in st.session_state.DATA.columns
            ],
            "Max": [
                st.session_state.DATA[col].max() for col in st.session_state.DATA.columns
            ],
            "Skew": [
                st.session_state.DATA[col].skew() for col in st.session_state.DATA.columns
            ],
            "Kurtosis": [
                st.session_state.DATA[col].kurtosis() for col in st.session_state.DATA.columns
            ]
        }
    )
    st.dataframe(df_summary, key="df_summary", hide_index=True)

    st.space("xxsmall")

    st.subheader("Hypothesis testing")
    st.caption("Central tendency")

    col_1, col_2, col_3, col_4, col_5 = st.columns(5)
    with col_1:
        parameter_to_test = st.selectbox(
            "Parameter of interest",
            options=["Mean", "Median"],
            index=0,
            key="test_param"
        )
    with col_2:
        variable_to_test = st.selectbox(
            "Variable to test",
            options=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])],
            index=0,
            key="test_var"
        )
    with col_3:
        direction_to_test = st.selectbox(
            "Direction",
            options=[">", "<", "≠"],
            index=0,
            key="value_direction"
        )
    with col_4:
        value_to_test = st.number_input(
            "Value",
            value=0.0,
            key="test_value",
        )
    with col_5:
        sig_level = st.selectbox(
            "Level of significance",
            options=[0.01, 0.05, 0.1],
            index=1,
            key="test_alpha"
        )

    alternatives = {
        ">": "greater",
        "<": "less",
        "≠": "two-sided"
    }

    test_stat = None
    p_val = None
    test_done = None

    if parameter_to_test == "Mean":
        test_result = stats.ttest_1samp(
            st.session_state.DATA[variable_to_test],
            value_to_test,
            alternative=alternatives[direction_to_test]
        )
        test_stat = test_result.statistic
        p_val = test_result.pvalue
        test_done = "one-sample t-test for the mean"
    elif parameter_to_test == "Median":
        test_stat, p_val = sign_test(st.session_state.DATA[variable_to_test], value_to_test,
                                     alternative=alternatives[direction_to_test])
        test_done = "sign test"

    test_stat = np.round(test_stat, 4)
    p_val = np.round(p_val, 4)
    decision = "enough" if p_val <= float(sig_level) else "not enough"

    if alternatives[direction_to_test] == "two-sided":
        claim_direction = "equal to"
    elif alternatives[direction_to_test] == "greater":
        claim_direction = "greater than"
    else:
        claim_direction = "less than"

    st.success(f"""
    Based on the **{test_done}**, the test statistic is {test_stat}, with a **p-value of {p_val}**. At {sig_level} 
    level of significance, there is **{decision} evidence** to support the claim that the population 
    {parameter_to_test.lower()} is {claim_direction} {value_to_test}.
    """)

    st.caption("Sphericity")
    col_1, col_2 = st.columns([4, 1])
    with col_1:
        variables_to_test = st.multiselect(
            "Variables to test",
            options=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])],
            key="test_vars_to_test",
            help="Note that this test applies only to Pearson correlations."
        )
    with col_2:
        sig_level = st.selectbox(
            "Level of significance",
            options=[0.01, 0.05, 0.1],
            index=1,
            key="test_alpha_2"
        )

    if len(variables_to_test) < 2:
        st.warning("Choose at least two variables.")
    else:
        test_stat, p_val = InterpretableFA(st.session_state.DATA[variables_to_test]).sphericity
        test_stat = np.round(test_stat, 4)
        p_val = np.round(p_val, 4)
        decision = "enough" if p_val <= float(sig_level) else "not enough"

        st.success(f"""
        Based on **Bartlett's test for sphericity**, the test statistic is {test_stat}, with a **p-value of {p_val}**. 
        At {sig_level} level of significance, there is {decision} evidence to reject the null hypothesis that the 
        correlation matrix is an identity matrix (i.e., all pairwise correlations are 0).
        """)

    st.space("xxsmall")

    st.subheader("Distribution")
    x_to_show = st.selectbox(
        "Manifest variable",
        options=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])],
        index=0
    )
    fig_x_dist = px.histogram(
        x=st.session_state.DATA[x_to_show],
        color_discrete_sequence=discrete_colors,
        labels={"x": "Value"},
        title=""
    )
    fig_x_dist.update_layout(
        margin=dict(t=20, b=20, r=20, l=20),
        height=300,
        bargap=0.1,
        yaxis=dict(
            title="Count",
            visible=True,
            showticklabels=True,
            showline=show_y_line,
            showgrid=show_y_grid,
            zeroline=False
        ),
        xaxis=dict(
            visible=True,
            showticklabels=True,
            showline=show_x_line,
            showgrid=show_x_grid,
            zeroline=False
        )
    )
    st.plotly_chart(fig_x_dist, width="stretch", key=f"basic_stats_dist_{x_to_show}")

    st.subheader("Correlation matrix")
    if st.session_state.LARGEST_POLY_CORR is None:
        st.info(":material/info: If you wish to use polychoric correlations, re-load the dataset and choose the "
                "option to calculate polychoric correlations.")
    stats_corr_type = st.selectbox(
        "Correlation type",
        options=["Pearson", "Polychoric"] if st.session_state.LARGEST_POLY_CORR is not None else ["Pearson"],
        index=0,
        key="stats_corr_type"
    )
    corr_mat = st.session_state.DATA.corr() if stats_corr_type == "Pearson" else st.session_state.LARGEST_POLY_CORR
    fig_corr = px.imshow(
        corr_mat,
        text_auto="0.3f",
        aspect="auto",
        color_continuous_scale=continuous_diverging_color_palette,
        zmin=-1, zmax=1,
        title=""
    )
    fig_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
    fig_corr.update_yaxes(tickmode="linear", dtick=1)
    fig_corr.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=min(900, max(50 * st.session_state.DATA.shape[1], 300))
    )
    st.plotly_chart(fig_corr, width="stretch", key="basic_stats_fig_corr")
    st.download_button("Download correlation matrix as CSV file", corr_mat.to_csv(),
                       file_name="corr_mat.csv", width="stretch")

    st.space("xxsmall")

    st.subheader("Semantic similarity matrix")
    if st.session_state.STATEMENTS is None:
        st.warning("Associated statements are not available / were not uploaded.")
    else:
        embeddings = st.session_state.EMBEDDINGS
        if embeddings is not None:
            semantic_similarity_mat = st.session_state.SEMANTIC_SIMILARITY_MATRIX
            fig_semantic = px.imshow(
                semantic_similarity_mat,
                text_auto="0.3f",
                aspect="auto",
                color_continuous_scale=continuous_sequential_color_palette,
                zmin=0, zmax=1,
                title=""
            )
            fig_semantic.update_xaxes(side="bottom", tickmode="linear", dtick=1)
            fig_semantic.update_yaxes(tickmode="linear", dtick=1)
            fig_semantic.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=min(900, max(50 * st.session_state.DATA.shape[1], 300))
            )
            st.plotly_chart(fig_semantic, width="stretch", key="basic_stats_fig_semantic")

            st.download_button("Download semantic similarity matrix as CSV file",
                               semantic_similarity_mat.to_csv(), file_name="semantic_sim_mat.csv", width="stretch")
        else:
            st.info("Loading...")

    st.space("xxsmall")


@st.dialog(":material/add_chart: Fit a new factor model", width="large", dismissible=False)
def fit_model_dialog():
    model_name = None
    number_of_factors = None
    corr_type = None
    estimation_method = None
    rotation = None
    prior = None
    prior_matrix = None
    manifest_vars = None

    if st.session_state.LARGEST_POLY_CORR is None:
        st.info(":material/info: If you wish to use polychoric correlations, re-load the dataset and choose the "
                "option to calculate polychoric correlations.")

    can_proceed = True

    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        model_name = st.text_input(label="Model name", value=f"model_{len(st.session_state.FACTOR_MODELS) + 1}",
                                   help="""
                                   This must be unique. It is recommended to be descriptive (e.g., "minres_varimax_4").
                                   """,
                                   width="stretch", key="model_name_fit")
    with col_2:
        number_of_factors = st.number_input(label="Number of factors", placeholder="Enter a positive number...",
                                            value=3, min_value=1, max_value=(st.session_state.DATA.shape[1] - 1),
                                            help="The number of factors cannot exceed the number of manifest "
                                                 "variables.", width="stretch", key="num_fac_input")
        factor_count_warning = st.empty()
    with col_3:
        corr_type = st.selectbox(
            "Type of correlation to use",
            options=["Pearson", "Polychoric"] if st.session_state.LARGEST_POLY_CORR is not None else ["Pearson"],
            index=0,
            key="corr_type_input",
            help="Polychoric correlation is generally considered to be more appropriate but it is computationally "
                 "more expensive."
        )

    manifest_vars = st.multiselect(
        "Manifest variables",
        options=[f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])],
        help="Only these manifest variables will be considered in the factor model.",
        key="manifest_fit"
    )

    if len(manifest_vars) >= 2 and get_df(len(manifest_vars), number_of_factors) < 0:
        factor_count_warning.warning("""
        Note that the number of factors is too large. The degrees of freedom is negative.
        """)

    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        estimation_method = st.selectbox(
            label="Estimation method", placeholder="Select an estimation method...",
            key="estimation_method_fit", options=sorted(list(ESTIMATION_METHODS.keys())),
            index=sorted(list(ESTIMATION_METHODS.keys())).index("Minimum Residual")
        )
    with col_2:
        rotation = st.selectbox(label="Rotation", placeholder="Select a rotation method...", key="rotation_fit",
                                options=ROTATIONS, help="Priorimax, Varimax, Oblimax, Quartimax, and Equamax are "
                                                        "orthogonal rotations. Promax, Oblimin, and Quartimin are "
                                                        "oblique rotations.", width="stretch")
    with col_3:
        prior = st.selectbox(label="Prior matrix", placeholder="Specify the prior matrix for priorimax...",
                             options=("Semantics", "Grouped", "Custom", "None"), key="prior_fit",
                             help="""
                             "Semantics" uses the semantic similarity matrix. "Custom" lets you specify the exact 
                             matrix by uploading a CSV file. This is required when using the priorimax rotation and 
                             optional for other rotation methods.""", width="stretch")
        show_prior_matrix = st.checkbox("Show exact prior matrix?", key="show_exact_prior_checkbox")

    if len(manifest_vars) < 2:
        st.warning("Please select at least two manfiest variables first.")
        can_proceed = False

    if can_proceed:
        if prior == "Semantics":
            if st.session_state.SEMANTIC_SIMILARITY_MATRIX is None:
                st.warning("You cannot use the semantic similarity matrix for the prior matrix since no statements "
                           "were loaded.")
                can_proceed = False
            if can_proceed:
                semantic_similarity_mat = st.session_state.SEMANTIC_SIMILARITY_MATRIX.loc[manifest_vars, manifest_vars]
                if show_prior_matrix:
                    fig_semantic = px.imshow(
                        semantic_similarity_mat,
                        text_auto="0.3f",
                        aspect="auto",
                        color_continuous_scale=continuous_sequential_color_palette,
                        zmin=0, zmax=1
                    )
                    fig_semantic.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                    fig_semantic.update_yaxes(tickmode="linear", dtick=1)
                    fig_semantic.update_layout(
                        height=min(900, max(50 * len(manifest_vars), 300)),
                        title="The semantic similarity matrix will be used as the prior matrix."
                    )
                    st.plotly_chart(fig_semantic, width="stretch", key="model_fit_fig_semantic")
                prior_matrix = semantic_similarity_mat
        elif prior == "Grouped":
            cols = st.columns(3)
            if st.session_state.STATEMENTS is not None:
                group_choices = [f"X{i + 1} - {st.session_state.STATEMENTS[i]}"
                                 for i in range(st.session_state.DATA.shape[1])]
            else:
                group_choices = [f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])]
            group_choices = [
                group_choice
                for group_choice in group_choices if group_choice.split(" - ")[0] in manifest_vars
            ]

            groupings = []
            if number_of_factors is not None:
                for factor_number in range(number_of_factors):
                    current_col = cols[factor_number % len(cols)]
                    with current_col:
                        groupings.append(
                            st.multiselect(label=f"Grouping {factor_number + 1}", options=group_choices,
                                           key=f"grouping_for_{factor_number}_fit")
                        )

            for idx, grouping in enumerate(groupings):
                if len(grouping) < 2:
                    st.warning(f"Grouping {idx + 1} must have at least two entries.")
                    can_proceed = False
                groupings[idx] = [group_choice.split(" - ")[0] for group_choice in grouping]

            if can_proceed:
                prior_matrix = np.zeros((len(manifest_vars), len(manifest_vars)), dtype=np.int8)
                np.fill_diagonal(prior_matrix, 1)
                prior_matrix = pd.DataFrame(prior_matrix, columns=manifest_vars, index=manifest_vars)
                for grouping in groupings:
                    for pair in combinations(grouping, 2):
                        prior_matrix.loc[pair[0], pair[1]] = 1
                        prior_matrix.loc[pair[1], pair[0]] = 1

                if show_prior_matrix:
                    fig_prior = px.imshow(
                        prior_matrix,
                        text_auto="0",
                        aspect="auto",
                        color_continuous_scale=continuous_sequential_color_palette,
                        zmin=0, zmax=1
                    )
                    fig_prior.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                    fig_prior.update_yaxes(tickmode="linear", dtick=1)
                    fig_prior.update_layout(
                        height=min(900, max(50 * prior_matrix.shape[1], 300)),
                        title="This grouping matrix will be used as the prior matrix."
                    )
                    st.plotly_chart(fig_prior, width="stretch", key="model_fit_grouped_prior")
        elif prior == "Custom":
            prior_matrix = st.file_uploader(label="Upload CSV file for the prior matrix", type="csv", width="stretch",
                                            disabled=(prior != "Custom"))
            if prior_matrix is not None:
                prior_matrix = pd.read_csv(prior_matrix, header=None).to_numpy()

                check = process_prior_matrix(prior_matrix, rotation, manifest_vars)
                if check["pass"]:
                    prior_matrix = check["processed_matrix"].copy()
                    prior_matrix = pd.DataFrame(prior_matrix, index=manifest_vars, columns=manifest_vars)
                    if show_prior_matrix:
                        fig_prior = px.imshow(
                            prior_matrix,
                            text_auto="0.3f",
                            aspect="auto",
                            color_continuous_scale=continuous_sequential_color_palette,
                            zmin=0, zmax=1
                        )
                        fig_prior.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                        fig_prior.update_yaxes(tickmode="linear", dtick=1)
                        fig_prior.update_layout(
                            height=min(900, max(50 * len(manifest_vars), 300)),
                            title="This custom matrix will be used as the prior matrix."
                        )
                        st.plotly_chart(fig_prior, width="stretch", key="custom_prior_fit")
                else:
                    st.error(check["message"])
                    can_proceed = False
            else:
                manifest_statements = st.session_state.STATEMENTS_DF[
                    st.session_state.STATEMENTS_DF["Variable"].isin(manifest_vars)
                ].sort_values(by="Variable", ascending=True, key=lambda x: x.map({
                    val: key
                    for key, val in enumerate(manifest_vars)
                })).reset_index(drop=True)
                st.warning("""
                Please upload the CSV file. Please make sure that the order of the rows (and columns) match the order of 
                the manifest variables shown below.
                """)
                st.dataframe(manifest_statements, key="man_stat_df")
                can_proceed = False
        elif prior == "None":
            prior_matrix = None

    st.space()
    _, col_2, col_3 = st.columns([12, 3, 3])
    with col_3:
        if st.button("Cancel", width="stretch", key="cancel_fit_btn"):
            st.rerun()
    if can_proceed:
        with col_2:
            if st.button("Fit model", width="stretch", type="primary", key="confirm_fit_btn"):
                any_failed = False

                if model_name.strip() == "" or model_name is None:
                    st.toast("🚫 The model name cannot be empty.", duration="long")
                    any_failed = True
                if model_name in st.session_state.FACTOR_MODELS.keys():
                    st.toast("🚫 The model name must be unique.")
                    any_failed = True

                try:
                    number_of_factors = int(number_of_factors)
                except ValueError:
                    st.toast("🚫 The number of factors must be an integer.", duration="long")
                    any_failed = True
                if number_of_factors < 1 or number_of_factors >= len(manifest_vars):
                    st.toast("🚫 The number of factors must be at least 1 but less than the number of "
                             "manifest variables.", duration="long")
                    any_failed = True

                if corr_type not in ["Pearson", "Polychoric"]:
                    st.toast("🚫 Please enter a valid correlation type.", duration="long")
                    any_failed = True

                if estimation_method not in ESTIMATION_METHODS:
                    st.toast("🚫 Please enter a valid estimation method.", duration="long")
                    any_failed = True

                if rotation not in ROTATIONS:
                    st.toast("🚫 Please enter a valid rotation method.", duration="long")
                    any_failed = True

                if rotation == "Priorimax" and prior_matrix is None:
                    st.toast("🚫 A prior matrix is required for priorimax.", duration="long")
                    any_failed = True

                if prior not in ["Semantics", "Grouped", "Custom", "None"]:
                    st.toast("🚫 The prior matrix must be based on semantics, groupings, or custom values",
                             duration="long")
                    any_failed = True

                if prior != "None":
                    check = process_prior_matrix(prior_matrix.to_numpy(), rotation, manifest_vars)
                    if not check["pass"]:
                        st.toast("🚫" + check["message"], duration="long")
                        any_failed = True
                    else:
                        prior_matrix = check["processed_matrix"].copy()
                else:
                    prior_matrix = None

                if not any_failed:
                    st.session_state.model_name = model_name
                    st.session_state.number_of_factors = number_of_factors
                    st.session_state.corr_type = corr_type
                    st.session_state.estimation_method = estimation_method
                    st.session_state.rotation = rotation
                    st.session_state.prior_matrix = prior_matrix
                    st.session_state.prior = prior
                    st.session_state.manifest_vars = manifest_vars
                    st.session_state.FIT_MODEL = "Yes"
                    st.rerun()


@st.dialog(":material/bar_chart: View factor models", width="large", on_dismiss="rerun")
def view_models_dialog():
    model_name = st.selectbox("Choose a model", options=sorted(list(st.session_state.FACTOR_MODELS.keys())),
                              width="stretch", key="choose_model_select")

    if model_name is None:
        st.error("There are no factor models available.")
    else:
        _, col_2 = st.columns([21, 3])
        with col_2:
            st.button("Delete model", on_click=delete_factor_model, args=(model_name,),
                      width="stretch", key="delete_model_btn")

        st.subheader("Fit details")

        factor_model = st.session_state.FACTOR_MODELS[model_name].models[model_name]
        fit_details = st.session_state.FIT_DETAILS[model_name]

        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            st.caption("Number of manifest variables")
            st.badge(str(len(fit_details["manifest_vars"])), color="green")
        with col_2:
            st.caption("Number of factors")
            st.badge(str(fit_details["number_of_factors"]), color="green")
        with col_3:
            st.caption("Correlation type")
            st.badge(str(fit_details["corr_type"]), color="green")

        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            st.caption("Estimation Method")
            st.badge(str(fit_details["estimation_method"]).capitalize(), color="green")
        with col_2:
            st.caption("Rotation")
            st.badge(str(fit_details["rotation"]).capitalize(), color="green")
        with col_3:
            st.caption("V-index")
            v = st.session_state.FACTOR_MODELS[model_name].calculate_v_index(model_name)
            v = np.round(v, 5) if v is not None else None
            st.badge(str(v), color="green")

        st.space()

        with st.expander("View correlation matrix", expanded=False):
            if fit_details["corr_type"] == "Pearson":
                df_corr = st.session_state.DATA[fit_details["manifest_vars"]].corr()
            else:
                df_corr = st.session_state.LARGEST_POLY_CORR.loc[
                    fit_details["manifest_vars"], fit_details["manifest_vars"]
                ]

            fig_corr = px.imshow(
                df_corr,
                text_auto="0.3f",
                aspect="auto",
                color_continuous_scale=continuous_diverging_color_palette,
                zmin=-1, zmax=1
            )
            fig_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
            fig_corr.update_yaxes(tickmode="linear", dtick=1)
            fig_corr.update_layout(
                height=min(900, max(50 * len(fit_details["manifest_vars"]), 300)),
                title=f"Correlation Matrix ({str(fit_details['corr_type'])})"
            )
            st.plotly_chart(fig_corr, width="stretch", key="view_corr_view_model")
            _, col_2, _ = st.columns(3)
            with col_2:
                st.download_button("Download correlation matrix as CSV file", df_corr.to_csv(),
                                   file_name=f"{model_name}_corr_matrix.csv", width="stretch")
            st.space()
        if fit_details["prior"] != "None":
            with st.expander("View prior matrix", expanded=False):
                df_prior_matrix = pd.DataFrame(factor_model.prior_,
                                               index=fit_details["manifest_vars"],
                                               columns=fit_details["manifest_vars"])
                fig_prior = px.imshow(
                    df_prior_matrix,
                    text_auto="0.3f",
                    aspect="auto",
                    color_continuous_scale=continuous_sequential_color_palette,
                    zmin=0, zmax=1
                )
                fig_prior.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                fig_prior.update_yaxes(tickmode="linear", dtick=1)
                fig_prior.update_layout(
                    height=min(900, max(50 * len(fit_details["manifest_vars"]), 300)),
                    title=f"Prior Matrix ({str(fit_details['prior']).capitalize()})"
                )
                st.plotly_chart(fig_prior, width="stretch", key="view_prior_view_model")
                _, col_2, _ = st.columns(3)
                with col_2:
                    st.download_button("Download prior matrix as CSV file", df_prior_matrix.to_csv(),
                                       file_name=f"{model_name}_prior_matrix.csv", width="stretch")
                st.space()
        else:
            st.warning("No prior matrix was used for this model.")
        st.space()

        st.subheader("Model details")
        model_analysis = st.session_state.FACTOR_MODELS[model_name].analyze_model(model_name).reset_index(drop=True)
        means = pd.DataFrame({
            "variable": [manifest_var for manifest_var in fit_details["manifest_vars"]],
            "mean": [st.session_state.DATA[manifest_var].mean() for manifest_var in fit_details["manifest_vars"]]
        })
        model_analysis = pd.merge(left=model_analysis, right=means, on="variable", how="inner")
        if st.session_state.STATEMENTS_DF is not None:
            model_analysis = pd.merge(left=model_analysis, right=st.session_state.STATEMENTS_DF,
                                      left_on="variable", right_on="Variable", how="inner")
            model_analysis = model_analysis[
                ["variable", "Statement", "mean"] +
                [col for col in model_analysis.columns if col.startswith("factor_")] +
                ["communality", "kmo_msa"]
                ]
        model_analysis.columns = [col.upper() for col in model_analysis.columns]
        model_analysis_styled = model_analysis.style.background_gradient(
            cmap=continuous_diverging_color_palette, axis=None,
            subset=[col for col in model_analysis.columns if col.startswith("FACTOR_")],
            vmin=-1.0, vmax=1.0
        ).background_gradient(
            cmap=(
                continuous_sequential_color_palette
                if continuous_sequential_color_palette not in COLOR_SPECIAL_SEQUENTIAL
                else continuous_sequential_color_palette.lower()
            ), axis=0, subset=["COMMUNALITY", "KMO_MSA"], vmin=0, vmax=1.0
        )

        st.caption("Means, correlations, communalities, and sampling adequacies")
        st.dataframe(model_analysis_styled, key="styled_comms_df")
        with st.expander("View interactive standardized loadings heatmap"):
            model_analysis = model_analysis.set_index("VARIABLE")
            model_analysis = model_analysis[[col for col in model_analysis.columns if col.startswith("FACTOR_")]]
            fig_loadings = px.imshow(
                model_analysis,
                text_auto="0.3f",
                aspect="auto",
                color_continuous_scale=continuous_diverging_color_palette,
                color_continuous_midpoint=0,
                labels=dict(x="Factors", y="Variables", color="Standardized loading"),
                title="Factor Loading Matrix (Standardized)"
            )
            fig_loadings.update_xaxes(side="bottom", tickmode="linear", dtick=1)
            fig_loadings.update_yaxes(tickmode="linear", dtick=1)
            fig_loadings.update_layout(
                height=min(900, max(50 * len(fit_details["manifest_vars"]), 300))
            )
            st.plotly_chart(fig_loadings, width="stretch", key="view_loadings_view_model")

        st.space()

        factor_scores = st.session_state.FACTOR_MODELS[model_name].models[model_name].transform(
            st.session_state.DATA
        )
        df_factor_scores = pd.DataFrame(factor_scores, columns=[f"factor_{idx + 1}"
                                                                for idx in range(factor_scores.shape[1])])
        df_factor_scores_long = df_factor_scores.melt(var_name="Factor", value_name="Score")
        fig_scores_hist = px.histogram(
            df_factor_scores_long,
            x="Score",
            color="Factor",
            marginal="box",
            barmode="overlay",
            opacity=0.8,
            color_discrete_sequence=discrete_colors
        )
        st.caption("Factor score distribution")
        st.plotly_chart(fig_scores_hist, width="stretch", key="view_factor_scores_view_model")
        df_data_with_factor_scores = pd.concat([st.session_state.DATA, df_factor_scores], axis=1)
        _, col_2, _ = st.columns(3)
        with col_2:
            st.download_button("Download factor scores as CSV file", df_data_with_factor_scores.to_csv(),
                               file_name=f"{model_name}_factor_scores.csv", width="stretch")

        st.space()

        if st.session_state.FACTOR_MODELS[model_name].models[model_name].is_orthogonal_:
            factor_corr_mat = np.eye(st.session_state.FIT_DETAILS[model_name]["number_of_factors"])
        else:
            factor_corr_mat = st.session_state.FACTOR_MODELS[model_name].models[model_name].phi_
        df_factor_corr_mat = pd.DataFrame(factor_corr_mat,
                                          columns=[f"factor_{idx + 1}" for idx in range(factor_scores.shape[1])],
                                          index=[f"factor_{idx + 1}" for idx in range(factor_scores.shape[1])])
        fig_factor_corr = px.imshow(
            df_factor_corr_mat,
            text_auto="0.3f",
            aspect="auto",
            color_continuous_scale=continuous_diverging_color_palette,
            zmin=-1, zmax=1,
        )
        fig_factor_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
        fig_factor_corr.update_yaxes(tickmode="linear", dtick=1)
        st.caption("Factor correlations")
        st.plotly_chart(fig_factor_corr, width="stretch", key="view_factor_corr_view_model")
        _, col_2, _ = st.columns(3)
        with col_2:
            st.download_button("Download factor correlations as CSV file", df_factor_corr_mat.to_csv(),
                               file_name=f"{model_name}_factor_correlations.csv", width="stretch")

        st.space()


# Loader
_, col_2, _ = st.columns([1, 3, 1])
with col_2:
    with open("./images/factor_flow_logo.png", "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <img class="hide-logo" src="data:image/png;base64,{encoded_logo}" />
    """, unsafe_allow_html=True)
    st.space()
    with st.container(horizontal_alignment="center"):
        with st.spinner("Loading NLP models...", show_time=True):
            while st.session_state.USE_MODEL_LOADED != 1:
                time.sleep(0.1)
if st.session_state.USE_MODEL_LOADED == 1:
    st.markdown("""
        <style>
            .hide-logo {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title(":material/menu: Menu")

with st.sidebar.expander("NLP Models", icon=":material/graph_3:", expanded=True):
    st.subheader("Universal Sentence Encoder")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.caption("Embedder:")
        if st.session_state.USE_MODEL_LOADED == 1:
            st.badge("Loaded", color="green")
        else:
            st.badge("Loading", color="yellow")
    with col_2:
        st.caption("Embeddings:")
        if isinstance(st.session_state.EMBEDDINGS, np.ndarray):
            st.badge("Calculated", color="green")
        else:
            if st.session_state.STATEMENTS is None:
                st.badge("No statements loaded", color="yellow")
            else:
                st.badge("Calculating", color="blue")

    st.subheader(f"Large Language Model")

    try:
        llms = st.session_state.LLM_CLIENT.models.list()
        connected = True
    except Exception as e:
        connected = False

    if connected:
        col_label, col_help = st.columns([0.9, 0.1])
        with col_label:
            st.caption("Model in use:")
        with col_help:
            st.markdown("", help="Different large language models may give different results. "
                                 "They also have different usage rate limits. "
                                 "Note that regardless of the model, the maximum completion tokens is 4096.")
        st.session_state.CURRENT_LLM_MODEL_ID = st.selectbox(
            "Model in use:",
            options=LLM_MODEL_IDS,
            index=0,
            label_visibility="collapsed",
            key="llm_choice"
        )

        col_label, col_help = st.columns([0.9, 0.1])
        with col_label:
            st.caption("LLM temperature:")
        with col_help:
            st.markdown("", help="Larger values encourage randomness and creativity, while "
                                 "smaller values encourage determinism and focus. For more consistent interpretations "
                                 "and formatting, choose a value not greater than 0.1.")
        llm_temp = st.slider("LLM temperature:", min_value=0.0, max_value=0.5, step=0.025, value=0.0,
                             label_visibility="collapsed", key="llm_temp_slider")
    else:
        st.error("Failed to connect. Please refresh.")

    if st.button("Reload models", type="secondary", width="stretch"):
        st.markdown(
            """
            <meta http-equiv="refresh" content="0">
            """,
            unsafe_allow_html=True
        )

with st.sidebar.expander("Data", icon=":material/dataset:", expanded=True):
    if st.session_state.DATA is None:
        st.warning("You have not uploaded a dataset yet.")
        _, col_2, _ = st.columns([1, 5, 1])
        with col_2:
            st.button("Upload", width="stretch", disabled=(st.session_state.USE_MODEL_LOADED != 1),
                      on_click=upload_data_dialog, key="upload_data_btn", type="primary")
    else:
        st.subheader("Dataset")
        st.text_input("Dataset", value=str(st.session_state.DATA_NAME),
                      label_visibility="collapsed", disabled=True, key="dataset_label_input")

        st.subheader("Statements")
        st.text_input("Statements name", value=str(st.session_state.STATEMENTS_NAME),
                      label_visibility="collapsed", disabled=True, key="statements_label_input")

        col_1, col_2 = st.columns(2)
        with col_1:
            st.caption("Variables")
            st.write(str(st.session_state.DATA.shape[1]))
        with col_2:
            st.caption("Observations")
            st.write(f"{st.session_state.DATA.shape[0]:,}")

        col_1, col_2 = st.columns(2)
        with col_1:
            st.caption("Is Likert-type?")
            if st.session_state.IS_LIKERT:
                st.write("Yes")
            elif st.session_state.IS_LIKERT is not None:
                st.write("No")
            else:
                st.write("N/A")
        with col_2:
            st.caption("Scale Direction")
            if st.session_state.LIKERT_DIRECTION == "01":
                st.write("Disagree-Agree")
            elif st.session_state.LIKERT_DIRECTION == "10":
                st.write("Agree-Disagree")
            else:
                st.write("N/A")

        col_1, col_2 = st.columns(2)
        with col_1:
            st.button("Stats", width="stretch", type="primary", key="stats_view_data", on_click=view_data_dialog)
        with col_2:
            st.button("Tags", width="stretch", key="view_tags_btn", on_click=view_tags_dialog)

        col_1, col_2 = st.columns(2)
        with col_1:
            st.button("Change", width="stretch", key="change_data_btn", on_click=upload_data_dialog)
        with col_2:
            if st.button("Clear", width="stretch", key="clear_data_btn"):
                st.session_state.DATA = None
                st.session_state.DATA_NAME = None
                st.session_state.LARGEST_POLY_CORR = None
                st.session_state.IS_LIKERT = None
                st.session_state.LIKERT_DIRECTION = None
                st.session_state.STATEMENTS = None
                st.session_state.STATEMENTS_DF = None
                st.session_state.EMBEDDINGS = None
                st.session_state.SEMANTIC_SIMILARITY_MATRIX = None
                st.session_state.FACTOR_MODELS = {}
                st.session_state.FIT_DETAILS = {}
                st.session_state.INTERPRETATIONS = {}
                st.session_state.VARIABLE_TAGS = {}
                clear_embeddings()
                st.rerun()

with st.sidebar.expander("Factor Models", icon=":material/function:", expanded=True):
    col_1, col_2 = st.columns(2)
    with col_1:
        st.button("Add", width="stretch", on_click=fit_model_dialog, type="primary", key="add_model_btn",
                  disabled=(st.session_state.USE_MODEL_LOADED != 1) or (st.session_state.DATA is None))
    with col_2:
        st.button("View", width="stretch", on_click=view_models_dialog, key="view_model_btn",
                  disabled=(st.session_state.DATA is None or len(st.session_state.FACTOR_MODELS) == 0))

    st.subheader("Models")
    if st.session_state.DATA is None or len(st.session_state.FACTOR_MODELS) == 0:
        st.warning("You have not estimated any factor model yet.")
    elif len(st.session_state.FACTOR_MODELS) > 0:
        model_names = sorted(list(st.session_state.FACTOR_MODELS.keys()))
        rotations = [st.session_state.FIT_DETAILS[model_name]["rotation"] for model_name in model_names]
        fact_counts = [st.session_state.FIT_DETAILS[model_name]["number_of_factors"] for model_name in model_names]
        var_counts = [len(st.session_state.FIT_DETAILS[model_name]["manifest_vars"]) for model_name in model_names]

        for i in range(len(model_names)):
            model_name = model_names[i]
            rotation = rotations[i]
            fact_count = fact_counts[i]
            var_count = var_counts[i]
            description = f"{fact_count}-factor {var_count}-variable {rotation if rotation is not None else ''}"

            st.caption(model_name)
            st.write(description)

with st.sidebar.expander("Settings", True, icon=":material/settings:"):
    st.caption("General")
    show_floating_top = st.checkbox("""Show "Back to Top" button""", key="show_back_to_top", value=True)
    font_size = st.slider("Choose the base font size (in px)", 12, 16, 14,
                          step=1, help="The default value is 14px.")

    if st.button("Reload app", type="secondary", width="stretch"):
        st.markdown(
            """
            <meta http-equiv="refresh" content="0">
            """,
            unsafe_allow_html=True
        )

    st.space()

    st.caption("Visualization colors")
    discrete_color_palette = st.selectbox("Discrete color palette",
                                          options=sorted(COLOR_DISCRETE_SCALES.keys()),
                                          index=list(sorted(COLOR_DISCRETE_SCALES.keys())).index("Plotly"),
                                          help="""
                                          This is the color palette that will be used for discrete labels. The default 
                                          is "Plotly". A typical colorblind-safe palette is "Safe".
                                          """)
    discrete_colors = COLOR_DISCRETE_SCALES[discrete_color_palette]
    fig_discrete_colors = px.imshow(
        [list(range(len(discrete_colors)))],
        color_continuous_scale=discrete_colors,
        aspect="auto"
    )
    fig_discrete_colors.update_layout(
        height=20,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        )
    )
    st.plotly_chart(
        fig_discrete_colors,
        width="stretch",
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )

    continuous_diverging_color_palette = st.selectbox(
        "Continuous diverging color palette",
        options=sorted(COLOR_CONTINUOUS_DIVERGING_SCALES.keys()),
        index=list(sorted(COLOR_CONTINUOUS_DIVERGING_SCALES.keys())).index("RdBu"),
        help="""
        This is the primary color palette that will be used for diverging continuous values (e.g., loadings). 
        The default is "RdBu". A typical colorblind-safe palette is "PuOr".
        """
    )
    continuous_diverging_colors = COLOR_CONTINUOUS_DIVERGING_SCALES[continuous_diverging_color_palette]
    fig_diverging_colors = px.imshow(
        [list(range(256))],
        color_continuous_scale=continuous_diverging_colors,
        aspect="auto"
    )
    fig_diverging_colors.update_layout(
        height=20,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        )
    )
    st.plotly_chart(
        fig_diverging_colors,
        width="stretch",
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )

    sequential_choices = list(
        set(COLOR_CONTINUOUS_SEQUENTIAL_SCALES.keys()) & set(plt.colormaps())
    )
    sequential_choices.extend(COLOR_SPECIAL_SEQUENTIAL)
    sequential_choices = sorted(sequential_choices)
    continuous_sequential_color_palette = st.selectbox(
        "Continuous sequential color palette",
        options=sequential_choices,
        index=sequential_choices.index("Viridis"),
        help="""
        This is the primary color palette that will be used for sequential continuous values (e.g., communalities). 
        The default is "Viridis"". A typical colorblind-safe palette is "Cividis".
        """
    )
    continuous_sequential_colors = COLOR_CONTINUOUS_SEQUENTIAL_SCALES[continuous_sequential_color_palette]
    fig_sequential_colors = px.imshow(
        [list(range(256))],
        color_continuous_scale=continuous_sequential_colors,
        aspect="auto"
    )
    fig_sequential_colors.update_layout(
        height=20,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        )
    )
    st.plotly_chart(
        fig_sequential_colors,
        width="stretch",
        config={
            "staticPlot": True,
            "displayModeBar": False
        }
    )

    st.space()

    st.caption("Chart axes and grid lines")
    show_x_line = not st.checkbox("Hide x-axis line", key="show_row_line", value=True)
    show_y_line = not st.checkbox("Hide y-axis line line", key="show_col_line", value=True)
    show_y_grid = not st.checkbox("Hide horizontal grid lines", key="show_row_grid", value=True)
    show_x_grid = not st.checkbox("Hide vertical grid lines", key="show_col_grid", value=True)


# Header
st.markdown(f"""
    <style>
        html {{
           font-size: {font_size}px;
        }}

        .stMainBlockContainer {{
            padding-top: 2rem !important;
            padding-bottom: 7rem !important;
        }}

        .st-key-load_use_model, .st-key-load_use_model iframe, .st-key-get_embeddings, .st-key-get_embeddings iframe {{
            height: 0px !important;
            background-color: rgba(0,0,0,0) !important;
        }}

        div:has(> iframe[title="streamlit_lottie.streamlit_lottie"]) {{
            overflow: hidden !important;
            margin: auto !important;
        }}

        .st-key-rotation_lottie iframe {{
            transform: scale(1) !important;
            transform-origin: center center !important;
        }}

        .custom-footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #FFFFFF;
            color: #7B8284;
            text-align: center;
            padding: 7px 30px 7px 30px;
            font-size: 11px;
            z-index: 999990;
        }}

        .feature-title {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}

        .feature-desc {{
            font-size: 1rem;
            line-height: 1.6;
        }}

        .feature-card {{
            display: flex;
            gap: 1rem;
            padding: 1rem 0;
            border-bottom: 2px solid #F0F0F0;
        }}

        .feature-icon {{
            min-width: 56px;
            border-radius: 16px;
            background: #F6FAFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }}

        .no-border {{
            border: none !important
        }}
    </style>

    <link href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined" rel="stylesheet">
""", unsafe_allow_html=True)


# Body
def overview_page():
    st.session_state.CURRENT_PAGE = "overview"

    _, col_2, _ = st.columns([1, 1.5, 1])
    with col_2:
        st.image("./images/factor_flow_logo.png", width="stretch")

    st.session_state.SCROLL_COUNTER = 1 - st.session_state.SCROLL_COUNTER
    with st.container(key=f"app_title_{st.session_state.SCROLL_COUNTER}"):
        _, col_2, _ = st.columns([1, 20, 1])
        with col_2:
            st.markdown(
                """
                <h5 style="text-align: center; color: #696969;">
                A Visual Analytics Workspace with Large Language Model–Assisted Interpretation for Factor Analysis
                </h5>
                """,
                unsafe_allow_html=True
            )

    st.space()

    with st.container(border=False):
        st.markdown("""
        ## :material/description: Description
        """)

        st.markdown("""
        **FactorFlow** is an interactive tool intended to help researchers and practitioners perform exploratory factor
        analysis (EFA) effectively and efficiently. Using this app, users can upload their datasets, perform
        diagnostics, fit various factor models and factor rotations, and evaluate models. It comes with the following
        key features:
        """)

        features = [
            ("<span class='material-icons-outlined'>calculate</span>", "Core exploratory factor analysis",
             "Classical estimation methods, rotations, and visualizations"),
            ("<span class='material-icons-outlined'>outbound</span>", "Beyond classical methods",
             "Pairwise target rotation and interpretability plots"
             "<a href='https://arxiv.org/abs/2409.11525' target='blank'><sup> learn more</sup></a>"),
            ("<span class='material-icons-outlined'>psychology</span>", "Large language model integration",
             "Language-model-assisted interpretation and semantic guidance"),
            ("<span class='material-icons-outlined'>analytics</span>", "Deep analysis workflows",
             "Hypothesis testing, diagnostics, model comparisons, and exploratory deep dives")
        ]

        col_1, col_2 = st.columns([2, 1])
        i = 0
        with col_1:
            for icon, title, desc in features:
                i += 1
                if i == len(features):
                    st.markdown(f"""
                        <div class="feature-card no-border">
                            <div class="feature-icon">{icon}</div>
                            <div>
                                <div class="feature-title">{title}</div>
                                <div class="feature-desc">{desc}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="feature-card">
                            <div class="feature-icon">{icon}</div>
                            <div>
                                <div class="feature-title">{title}</div>
                                <div class="feature-desc">{desc}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        with col_2:
            st.space("xxsmall")

            st_lottie(load_lottie_file("./lotties/rotation.json"),
                      speed=2, reverse=False, loop=True, quality="low", height=300, key="rotation_lottie")

        st.space("xsmall")

        col_1, col_2 = st.columns([2, 1])
        with col_1:
            st.button("See use cases and sample outputs", width="stretch", type="primary", on_click=use_cases_dialog)
        with col_2:
            st.button("Read the paper", width="stretch", type="secondary", disabled=True, help="Coming soon...")
        st.space("xxsmall")

    st.space()

    with st.container(border=False):
        st.markdown("""
        ## :material/rocket_launch: Getting started
        """)
        st.info("""
        For several widgets, you can hover at the :material/help: icon to see additional tips.
        """)

        st.markdown("""
        In general, you can follow these steps to use FactorFlow.
        """)

        getting_started_step = card_selector(
            [
                dict(
                    icon=":material/build:",
                    title="Configure (Optional)",
                    description="Set your preferences (or stick with the defaults)."
                ),
                dict(
                    icon=":material/upload:",
                    title="1. Upload",
                    description="Import your files.",
                ),
                dict(
                    icon=":material/feature_search:",
                    title="2. Explore",
                    description="Do some basic stats and diagnostics.",
                ),
                dict(
                    icon=":material/model_training:",
                    title="3. Fit",
                    description="Estimate factor models.",
                ),
                dict(
                    icon=":material/stacked_bar_chart:",
                    title="4 Analyze",
                    description="Examine your models.",
                ),
                dict(
                    icon=":material/export_notes:",
                    title="5. Export",
                    description="Download the results.",
                )
            ],
            default=1,
            key="how_to_guide"
        )

        st.button("Want more detailed instructions?", width="stretch", type="primary", on_click=guide_dialog)

        if getting_started_step == 0:
            st.markdown("""
            In the **Settings** panel under **Menu**, you can configure various settings for the tool, such as 
            the chart syles and colors. You can also click "Reload app" if you want to reset everything to their 
            defaults and clear all app data.
            """)
        elif getting_started_step == 1:
            st.markdown("""
            Upload your dataset in the **Data** section of the **Menu**. You can upload three kinds of files: 
            """)
            upload_sub_tab_1, upload_sub_tab_2, upload_sub_tab_3 = st.tabs([
                "A. Main dataset",
                "B. Statements",
                "C. Tags"
            ])

            with upload_sub_tab_1:
                st.markdown("""
                ###### The main dataset (CSV file). 
                This is the tabular dataset on which the factor models will be fit. Each 
                column must represent a feature and each observation must represent an observation. All data values 
                must be numeric and there must have no missing values. This is **required** to fit a model. The CSV file 
                or raw dataset **should not** have column headers. The tool will automatically label the columns as 
                X1, X2, and so on.

                A sample CSV file can be found 
                [here](https://drive.google.com/file/d/1NE02MwevCcHn4HXwxyXfF9AUjnSs5WGo/view?usp=sharing).
                """)

            with upload_sub_tab_2:
                st.markdown("""
                ###### The statements associated with the features (TXT file).
                This is the list of questions or statements associated with each feature in the main dataset. It must 
                be a text file, where statements are separated by linebreaks - consecutive lines with one statement 
                per line. The order of the statements must match the order of the columns in the main dataset (i.e., 
                the first statement must correspond to the first feature). This is an **optional** input, and will be 
                used only if you select "semantics" for the prior in pairwise target rotation.

                Ideally, statements should not be too long but they should also be "complete" (e.g., 
                a full sentence). However, it is also possible to have just "regular" one or two-word variable names 
                such as "height" and so on. In such cases though, semantic similarities may not be as meaningful.

                A sample TXT file can be found 
                [here](https://drive.google.com/file/d/1oXX7HNPhaOOvOTg_NgLV-Fla6O8EoQQi/view?usp=drive_link).
                """)

            with upload_sub_tab_3:
                st.markdown("""
                ###### The tags associated with each variable (manual or CSV file). 
                You can add tags to each variable or statement to help visualize interpretability. 
                To do so, click **Tags** under **Data**. Then, either upload manually or an appropriate CSV file. 
                This is optional.

                The CSV file must have two columns. The first column must contain the variables 
                and the second column must contain the corresponding tags. If a variable has 
                multiple tags, separate them using commas (e.g., "Tag A, Tag B"). If a variable 
                has no tags, leave the cell blank. The CSV file must not have headers.

                Sample CSV files can be found 
                [here](https://drive.google.com/file/d/1wVspesiKovOf_ZjrlEvXMs5BM-r4EN7F/view?usp=sharing) and 
                [here](https://drive.google.com/file/d/1JxMdwDOSoV0N5IPQKygeOBfVcnciLD2b/view?usp=sharing).

                **Statements vs Tags**. A variable can have at most one statement. It is usually the "question" for the 
                variable. No two variables can have the same statement. On the other hand, a variable can have zero or 
                more tags, and tags do not have to be unique across variables. Tags can be thought of as your 
                "initial" labels for the manifest variables.
                """)
        elif getting_started_step == 2:
            st.markdown("""
            Explore your dataset by going to the **Diagnostics** tab. For instance, you might want to examine the 
            communalities or you might want to determine the optimal number of factors. There is also an interactive 
            visualizer available if you want to manually explore the raw dataset yourself. Otherwise, you can go to 
            **Stats** under **Data** in the menu to see some readily available summary statistics.
            """)
        elif getting_started_step == 3:
            st.markdown("""
            Fit one or more factor models in the **Models** section of the **Menu**. Each model will use the same main 
            dataset. You can add or remove as many factor models as you need to. You can click the model name in order 
            to see more details about how the model was fit (e.g., number of factors, rotation method, fitting 
            algorithm).
            * When uploading a CSV file for a custom prior matrix, make sure that the matrix is symmetric and that 
            the number of rows (or columns) matches the number of manifest variables (i.e., columns in the main 
            dataset). Also, all entries must be either a number or left blank.
            * Note that the tool standardizes (i.e., subtracts the mean and divides by standard deviation) each manifest 
            variable prior to fitting. This means that the loadings provided are standardized loadings (i.e., 
            **correlations** with the factors).
            """)
        elif getting_started_step == 4:
            st.markdown("""
            Proceed to the **Dashboard** page and examine the loadings and visualizations available for each model. You 
            can choose to display only one model to focus on a single factor model but you can also display 2 factor 
            models at the same time for comparisons. If you want to view one factor model at a time in detail instead, 
            you can go to **View** under **Factor Models**.
            """)
        elif getting_started_step == 5:
            st.markdown("""
            You can **download** most tables, figures, and visualizations in this tool:
            * For tables, hovering on them triggers a download button to show at the top right of the table.
            * For most figures, there are dedicated download buttons that you can click.
            * For most visualizations, hovering on them triggers the control panel to show at the top right of the 
            chart, where you can find the download button. For some others, you can right-click on the chart and 
            click "Save image as..." or "Copy image".
            """)

        st.space("xxsmall")

    st.space()

    with st.container(border=False):
        st.markdown("""
        ## :material/pinboard: Notes
        """)

        st.markdown("""
        ###### Examples 
        * Sample datasets and files are available 
        [here](https://drive.google.com/drive/folders/1nc-pZFM5JdxmMrqE_QJyf03DLTEoEH0X?usp=sharing).
        * Classical rotations and traditional visualizations are made available here, but this tool was made partially 
        to make pairwise target rotation accessible, as well. As such, you may want to read the paper 
        [here](https://arxiv.org/abs/2409.11525) to understand more about how you can use this tool.

        ###### Future Releases
        * A video walkthrough of the tool is in the works.
        * The Universal Sentence Encoder is the only sentence embedder available for now.
        * Currently, support for confirmatory factor analysis (CFA) is not available but may be implemented in the 
        future.
        """)

        st.space("xxsmall")


def diagnostics_page():
    st.session_state.CURRENT_PAGE = "diagnostics"

    st.session_state.SCROLL_COUNTER = 1 - st.session_state.SCROLL_COUNTER
    with st.container(key=f"app_title_{st.session_state.SCROLL_COUNTER}"):
        st.write("")

    if st.session_state.DATA is None:
        st.warning("Upload a dataset first.")
    else:
        manifest_vars = st.multiselect(
            "Manifest variables",
            options=[f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])],
            help="Only these manifest variables will be considered in the diagnostics of the factor model.",
            key="choose_manifest_diag"
        )
        number_of_factors = st.slider(label="Number of factors", step=1,
                                      value=3, min_value=1, max_value=(st.session_state.DATA.shape[1] - 1),
                                      help="The number of factors cannot exceed the number of manifest "
                                           "variables.", width="stretch", key="num_fac_slider")

        st.space()

        if len(manifest_vars) < 2:
            st.warning("Select at least two manifest variables.")
        elif len(manifest_vars) <= number_of_factors:
            st.warning("The number of factors must be less than the number of manifest variables.")
        else:
            data_subset = st.session_state.DATA[manifest_vars]
            ifa = InterpretableFA(data_subset)
            ifa.fit_factor_model("model", number_of_factors, None)

            model_analysis = ifa.analyze_model("model").reset_index(drop=True)
            means = pd.DataFrame({
                "variable": [manifest_var for manifest_var in manifest_vars],
                "mean": [st.session_state.DATA[manifest_var].mean() for manifest_var in manifest_vars]
            })
            model_analysis = pd.merge(left=model_analysis, right=means, on="variable", how="inner")
            if st.session_state.STATEMENTS_DF is not None:
                model_analysis = pd.merge(left=model_analysis, right=st.session_state.STATEMENTS_DF,
                                          left_on="variable", right_on="Variable", how="inner")
                model_analysis = model_analysis[
                    ["variable", "Statement", "mean"] +
                    [col for col in model_analysis.columns if col.startswith("factor_")] +
                    ["communality", "kmo_msa"]
                    ]
            model_analysis.columns = [col.upper() for col in model_analysis.columns]
            model_analysis_styled = model_analysis[["VARIABLE", "STATEMENT", "COMMUNALITY", "KMO_MSA"]].copy()
            model_analysis_styled.sort_values(by=["COMMUNALITY"], ascending=[True], inplace=True)
            model_analysis_styled = model_analysis_styled.style.background_gradient(
                cmap=(
                    continuous_sequential_color_palette
                    if continuous_sequential_color_palette not in COLOR_SPECIAL_SEQUENTIAL
                    else continuous_sequential_color_palette.lower()
                ), axis=0, subset=["COMMUNALITY", "KMO_MSA"], vmin=0, vmax=1.0
            )

            with st.expander("Goodness-of-fit"):
                chisq_null, _ = get_chi_sq(ifa.models["model"], data_subset.shape[0], True)
                chisq_model, p_val = get_chi_sq(ifa.models["model"], data_subset.shape[0])
                df_null = get_df(data_subset.shape[1], 0)
                df_model = get_df(data_subset.shape[1], number_of_factors)

                if df_model < 0:
                    st.error(f"The number of factors is too large, with computed degrees of "
                             f"freedom {np.round(df_model, 4)}.")
                else:
                    st.info("""
                    :material/info: Note that the p-value is applicable only when maximum likelihood estimation 
                    is used.
                    """)

                    if pd.isna(chisq_model):
                        chisq_model = "Not Applicable"
                        cfi = "Not Applicable"
                        tli = "Not Applicable"
                        rmsea = "Not Applicable"
                    else:
                        cfi = 1 - (chisq_model - df_model) / (chisq_null - df_null)
                        tli = (chisq_null / df_null - chisq_model / df_model) / (chisq_null / df_null - 1)
                        rmsea = np.sqrt(max(chisq_model - df_model, 0) / (df_model * (data_subset.shape[0] - 1)))

                        cfi = str(np.round(cfi, 4)) if not pd.isna(cfi) else "Not Applicable"
                        tli = str(np.round(tli, 4)) if not pd.isna(tli) else "Not Applicable"
                        rmsea = str(np.round(rmsea, 4)) if not pd.isna(rmsea) else "Not Applicable"

                        chisq_model = str(np.round(chisq_model, 4))
                        p_val = str(np.round(p_val, 4))

                    col_1, col_2, col_3 = st.columns(3)
                    with col_1:
                        st.caption("Chi-squared statistic")
                        st.write(chisq_model)
                    with col_2:
                        st.caption("p-value")
                        st.write(p_val)
                    with col_3:
                        st.caption("Comparative Fit Index")
                        st.write(cfi)

                    col_1, col_2, _ = st.columns(3)
                    with col_1:
                        st.caption("Tucker-Lewis Index")
                        st.write(tli)
                    with col_2:
                        st.caption("Root Mean Square Error of Approximation")
                        st.write(rmsea)

            with st.expander("Communalities and adequacies"):
                st.dataframe(model_analysis_styled, key="styled_analysis_df")

            with st.expander("Scree plot"):
                loadings_only = model_analysis[[col for col in model_analysis.columns
                                                if col.startswith("FACTOR_")]].copy()
                squared_loadings = loadings_only ** 2
                eigenvalues = squared_loadings.sum().sort_values(ascending=False)
                df_eigenvalues = pd.DataFrame({
                    "Factor": eigenvalues.index,
                    "Sum of Squared Loadings": eigenvalues
                }).reset_index(drop=True)
                fig_scree = px.line(
                    df_eigenvalues,
                    x="Factor",
                    y="Sum of Squared Loadings",
                    text="Sum of Squared Loadings",
                    color_discrete_sequence=discrete_colors,
                    markers=True,
                    title="Eigenvalue per factor"
                )
                fig_scree.update_layout(
                    yaxis=dict(
                        visible=True,
                        showticklabels=True,
                        showline=show_y_line,
                        showgrid=show_y_grid,
                        zeroline=False
                    ),
                    xaxis=dict(
                        visible=True,
                        showticklabels=True,
                        showline=show_x_line,
                        showgrid=show_x_grid,
                        zeroline=False
                    )
                )
                fig_scree.update_traces(
                    mode="lines+markers+text",
                    texttemplate="%{text:.3f}",
                    textposition="top center"
                )
                fig_scree.add_hline(y=1, line_dash="dash", line_color=discrete_colors[1],
                                    annotation_text="Kaiser Criterion",
                                    annotation_position="top left")
                st.markdown(f"""
                The total sum of eigenvalues is 
                :blue-badge[{np.round(df_eigenvalues["Sum of Squared Loadings"].sum(), 4)}] out 
                of the theoretical maximum of :blue-badge[{len(manifest_vars)}].
                """)
                st.plotly_chart(fig_scree, width="stretch", key="fig_scree")
                st.space()

            with st.expander("Correlations"):
                if st.session_state.LARGEST_POLY_CORR is None:
                    st.info(
                        ":material/info: If you wish to use polychoric correlations, re-load the dataset and "
                        "choose the option to calculate polychoric correlations."
                    )

                diagnostics_corr_type = st.selectbox(
                    "Correlation type",
                    options=(["Pearson", "Polychoric"] if st.session_state.LARGEST_POLY_CORR is not None
                             else ["Pearson"]),
                    index=0,
                    key="diagnostics_corr_type"
                )

                final_corr_type = "Pearson"
                if diagnostics_corr_type == "Pearson":
                    corr_mat = data_subset.corr()
                    _, p_val = InterpretableFA(st.session_state.DATA[manifest_vars]).sphericity
                    p_val = np.round(p_val, 4)
                    st.success(f"""
                    The p-value for Bartlett's test for sphericity is {p_val}.
                    """)
                else:
                    if st.session_state.LARGEST_POLY_CORR is not None:
                        corr_mat = st.session_state.LARGEST_POLY_CORR.loc[manifest_vars, manifest_vars]
                        final_corr_type = "Polychoric"

                fig_corr = px.imshow(
                    corr_mat,
                    text_auto="0.2f",
                    aspect="auto",
                    color_continuous_scale=continuous_diverging_color_palette,
                    zmin=-1, zmax=1,
                    title=f"Subsetted correlation matrix ({final_corr_type})"
                )
                fig_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                fig_corr.update_yaxes(tickmode="linear", dtick=1)
                fig_corr.update_layout(
                    margin=dict(t=35, b=20, r=20, l=20),
                    height=min(650, max(25 * st.session_state.DATA.shape[1], 300))
                )
                st.plotly_chart(fig_corr, width="stretch", key="fig_corr_diag")


def dashboard_page():
    st.session_state.CURRENT_PAGE = "dashboard"

    st.session_state.SCROLL_COUNTER = 1 - st.session_state.SCROLL_COUNTER
    with st.container(key=f"app_title_{st.session_state.SCROLL_COUNTER}"):
        st.write("")

    if st.session_state.DATA is None or len(st.session_state.FACTOR_MODELS) == 0:
        st.warning("Fit a factor model first.")
    else:
        st.info("""
        :material/info: If your screen is not wide enough for the horizontal layout, consider temporarily hiding the 
        **Menu** sidebar. You can also hide or show columns in tables.
        """)

        col_1, _ = st.columns(2)
        with col_1:
            selected_models = st.multiselect(
                "Select models to examine", options=sorted(list(st.session_state.FACTOR_MODELS.keys())),
                help="You can choose up to 2 models at a time.",
                max_selections=2,
                key="compare_models_select"
            )

        st.space()
        if len(selected_models) == 0:
            st.warning("Choose at least one model.")
        else:
            model_analyses = [
                st.session_state.FACTOR_MODELS[model_name].analyze_model(model_name)
                for model_name in selected_models
            ]
            prior_matrices = [
                st.session_state.FACTOR_MODELS[model_name].models[model_name].prior_
                for model_name in selected_models
            ]
            loading_similarity_matrices = [
                st.session_state.FACTOR_MODELS[model_name].calculate_loading_similarity(model_name)
                for model_name in selected_models
            ]

            # Title
            cols = st.columns(len(selected_models), vertical_alignment="center")
            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]
                    st.header(model_name)
                    st.space()

            # Fit details
            cols = st.columns(len(selected_models), border=True)
            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]

                    st.subheader(":material/lists: Fit details")
                    st.space()
                    fit_details = st.session_state.FIT_DETAILS[model_name]

                    st.caption("QUANTITATIVE")

                    col_1, col_2, col_3, _ = st.columns(4)
                    col_1.metric("Variables", str(len(fit_details["manifest_vars"])))
                    col_2.metric("Factors", str(fit_details["number_of_factors"]))
                    v = st.session_state.FACTOR_MODELS[model_name].calculate_v_index(model_name)
                    v = np.round(v, 5) if v is not None else None
                    col_3.metric("V-Index", str(v))

                    st.space()

                    col_4, col_5, col_6, col_7 = st.columns(4)
                    with col_4:
                        st.caption("CORRELATION TYPE")
                        st.write(str(fit_details["corr_type"].capitalize()))
                    with col_5:
                        st.caption("ESTIMATION METHOD")
                        st.write(str(fit_details["estimation_method"].capitalize()))
                    with col_6:
                        st.caption("ROTATION")
                        st.write(str(fit_details["rotation"]).capitalize())
                    with col_7:
                        st.caption("PRIOR TYPE")
                        st.write(str(fit_details["prior"]))

                    st.space()

                    if st.session_state.FACTOR_MODELS[model_name].models[model_name].is_orthogonal_:
                        factor_corr_mat = np.eye(st.session_state.FIT_DETAILS[model_name]["number_of_factors"])
                    else:
                        factor_corr_mat = st.session_state.FACTOR_MODELS[model_name].models[model_name].phi_
                    df_factor_corr_mat = pd.DataFrame(factor_corr_mat,
                                                      columns=[f"factor_{idx + 1}" for idx in
                                                               range(st.session_state.FIT_DETAILS[model_name]
                                                                     ["number_of_factors"])],
                                                      index=[f"factor_{idx + 1}" for idx in
                                                             range(st.session_state.FIT_DETAILS[model_name]
                                                                   ["number_of_factors"])])
                    fig_factor_corr = px.imshow(
                        df_factor_corr_mat,
                        text_auto="0.3f",
                        aspect="auto",
                        color_continuous_scale=continuous_diverging_color_palette,
                        zmin=-1, zmax=1,
                    )
                    fig_factor_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                    fig_factor_corr.update_yaxes(tickmode="linear", dtick=1)
                    fig_factor_corr.update_layout(
                        margin=dict(t=0, b=0, r=0, l=0)
                    )
                    st.caption("FACTOR CORRELATIONS")
                    st.plotly_chart(fig_factor_corr, width="stretch", key=f"{model_name}_factor_corr_dashboard")

                    st.space("medium")

            # Communalities and adequacies
            cols = st.columns(len(selected_models), border=True)
            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]
                    model_analysis = model_analyses[i]

                    st.subheader(":material/monitoring: Communalities and adequacies")
                    st.space()

                    comm_and_adeq = model_analysis[["variable", "communality", "kmo_msa"]]
                    if st.session_state.STATEMENTS_DF is not None:
                        comm_and_adeq = pd.merge(left=comm_and_adeq, right=st.session_state.STATEMENTS_DF,
                                                 left_on="variable", right_on="Variable", how="left")
                        comm_and_adeq = comm_and_adeq[["variable", "Statement", "communality", "kmo_msa"]]
                    comm_and_adeq.columns = [col.upper() for col in comm_and_adeq.columns]
                    comm_and_adeq.sort_values(by=["COMMUNALITY"], ascending=[True], inplace=True)
                    comm_and_adeq_styled = comm_and_adeq.reset_index(drop=True)
                    comm_and_adeq_styled = comm_and_adeq_styled.style.background_gradient(
                        cmap=(
                            continuous_sequential_color_palette
                            if continuous_sequential_color_palette not in COLOR_SPECIAL_SEQUENTIAL
                            else continuous_sequential_color_palette.lower()
                        ), axis=0, subset=["COMMUNALITY", "KMO_MSA"], vmin=0,
                        vmax=1.0
                    )
                    st.dataframe(comm_and_adeq_styled, hide_index=True, key=f"{model_name}_comm_and_adeq")

                    st.space()

            # Interpretability plot
            cols = st.columns(len(selected_models), border=True)
            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]

                    st.subheader(":material/psychology: Interpretability plot")
                    st.space()

                    similarity_type = ("Semantic Similarity"
                                       if st.session_state.FIT_DETAILS[model_name]["prior"] == "Semantics"
                                       else "Prior Similarity")

                    if st.session_state.FIT_DETAILS[model_name]["prior"] == "None":
                        st.warning("No prior matrix was supplied for this model.")
                    else:
                        prior_matrix = prior_matrices[i]
                        loading_similarity_matrix = loading_similarity_matrices[i]

                        all_vars = st.session_state.FIT_DETAILS[model_name]["manifest_vars"]

                        df_prior_matrix = pd.DataFrame(
                            prior_matrix,
                            columns=all_vars,
                            index=all_vars
                        )

                        df_loading_sim_matrix = pd.DataFrame(
                            loading_similarity_matrix,
                            columns=all_vars,
                            index=all_vars
                        )

                        worst_var = None
                        best_v = 0
                        for manifest_var in all_vars:
                            curr_vars = all_vars.copy()
                            curr_vars.remove(manifest_var)
                            curr_prior_matrix = df_prior_matrix[curr_vars].to_numpy()
                            curr_loading_sim_matrix = df_loading_sim_matrix[curr_vars].to_numpy()
                            curr_v_index = get_v_index(curr_prior_matrix, curr_loading_sim_matrix)

                            if curr_v_index > best_v:
                                worst_var = manifest_var
                                best_v = curr_v_index

                        p, l, v_1, v_2 = get_multiset(prior_matrix, loading_similarity_matrix, True)
                        c = []
                        for pair in range(len(v_1)):
                            if v_1[pair] == worst_var or v_2[pair] == worst_var:
                                c.append(f"From {worst_var}")
                            else:
                                c.append(f"Not from {worst_var}")

                        df_multiset = pd.DataFrame({
                            similarity_type: p,
                            "Loading Similarity": l,
                            "Source": c
                        })

                        v_index = st.session_state.FACTOR_MODELS[model_name].calculate_v_index(model_name)
                        gap = np.round(best_v - v_index, 4)

                        st.caption(f"""
                        The V-index is about {np.round(v_index, 4)}. Out of all manifest variables, 
                        {worst_var} has the most inconsistent loadings relative to the expected similarities.
                        """)

                        show_worst = st.checkbox("Show inconsistent points?", key=f"{model_name}_show_inconsistent")

                        lowess_failed = False
                        try:
                            with warnings.catch_warnings(record=True) as w:
                                if show_worst:
                                    fig_v_plot = px.scatter(
                                        df_multiset,
                                        x=similarity_type,
                                        y="Loading Similarity",
                                        color="Source",
                                        trendline="lowess",
                                        color_discrete_sequence=discrete_colors,
                                        opacity=0.5,
                                        category_orders={
                                            "Source": [f"Not from {worst_var}", f"From {worst_var}"]
                                        },
                                        title=f"{similarity_type} vs Loading Similarity"
                                    )
                                else:
                                    fig_v_plot = px.scatter(
                                        df_multiset,
                                        x=similarity_type,
                                        y="Loading Similarity",
                                        trendline="lowess",
                                        color_discrete_sequence=discrete_colors,
                                        opacity=0.5,
                                        title=f"{similarity_type} vs Loading Similarity"
                                    )

                                if any("invalid value encountered in divide" in str(warn.message) for warn in w):
                                    raise RuntimeWarning("Lowess failed to fit.")
                        except RuntimeWarning:
                            if show_worst:
                                fig_v_plot = px.scatter(
                                    df_multiset,
                                    x=similarity_type,
                                    y="Loading Similarity",
                                    color="Source",
                                    trendline="ols",
                                    color_discrete_sequence=discrete_colors,
                                    opacity=0.5,
                                    category_orders={
                                        "Source": [f"Not from {worst_var}", f"From {worst_var}"]
                                    },
                                    title=f"{similarity_type} vs Loading Similarity"
                                )
                            else:
                                fig_v_plot = px.scatter(
                                    df_multiset,
                                    x=similarity_type,
                                    y="Loading Similarity",
                                    trendline="ols",
                                    color_discrete_sequence=discrete_colors,
                                    opacity=0.5,
                                    title=f"{similarity_type} vs Loading Similarity"
                                )

                            lowess_failed = True

                        fig_v_plot.update_layout(
                            yaxis=dict(
                                visible=True,
                                showticklabels=True,
                                showline=show_y_line,
                                showgrid=show_y_grid,
                                zeroline=False
                            ),
                            xaxis=dict(
                                visible=True,
                                showticklabels=True,
                                showline=show_x_line,
                                showgrid=show_x_grid,
                                zeroline=False
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.3,
                                xanchor="center",
                                x=0.2
                            )
                        )
                        fig_v_plot.update_traces(line=dict(width=3))
                        st.plotly_chart(fig_v_plot, width="stretch", key=f"{model_name}_v_plot")
                        if lowess_failed:
                            st.warning("Lowess failed to fit. Defaulted to OLS.")

                    st.space()

            # Factor breakdown
            cols = st.columns(len(selected_models), border=True)
            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]
                    model_analysis = model_analyses[i]
                    loadings_only = model_analysis[["variable"] + [col for col in model_analysis.columns
                                                                   if col.startswith("factor_")]]

                    st.subheader(":material/donut_small: Factor breakdown")
                    st.space()
                    show_breakdown_values = st.checkbox("Show value per tag?", value=True,
                                                        key=f"{model_name}_show_value_per_tag")
                    df_tags_breakdown = compute_tags_breakdown(loadings_only)
                    fig_tags_breakdown = px.bar(
                        df_tags_breakdown,
                        x="Factor",
                        y="Sum of Squared Loadings",
                        text="Sum of Squared Loadings",
                        color="Tag",
                        color_discrete_sequence=discrete_colors,
                        barmode="stack",
                        title="Sum of Squared Loadings per Factor"
                    )
                    fig_tags_breakdown.update_layout(
                        yaxis=dict(
                            visible=True,
                            showticklabels=False,
                            showline=show_y_line,
                            showgrid=show_y_grid,
                            zeroline=False
                        ),
                        xaxis=dict(
                            visible=True,
                            showticklabels=True,
                            showline=show_x_line,
                            showgrid=show_x_grid,
                            zeroline=False
                        )
                    )
                    if show_breakdown_values:
                        fig_tags_breakdown.update_traces(textposition="inside", texttemplate="%{text:.3f}",
                                                         insidetextanchor="middle")
                    else:
                        fig_tags_breakdown.update_traces(textposition="none")

                    totals = df_tags_breakdown.groupby("Factor")["Sum of Squared Loadings"].sum()
                    for factor, total in totals.items():
                        fig_tags_breakdown.add_annotation(
                            x=factor,
                            y=total,
                            text=f"{total:.3f}",
                            showarrow=False,
                            yshift=10,
                            xanchor="center",
                            yanchor="bottom"
                        )
                    st.plotly_chart(fig_tags_breakdown, width="stretch", key=f"{model_name}_tags_breakdown")

                    st.space()

            # Factor loadings
            cols = st.columns(len(selected_models), border=True)
            threshs = []
            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]
                    model_analysis = model_analyses[i]
                    loadings_only = model_analysis[["variable"] + [col for col in model_analysis.columns
                                                                   if col.startswith("factor_")]]

                    st.subheader(":material/bar_chart_4_bars: Factor loadings")
                    st.space()

                    show_each_factor = st.checkbox(
                        "Show the ECDF of the absolute loadings for each factor",
                        key=f"{model_name}_show_each_factor",
                        value=True,
                        help="""
                        If checked, the ECDF of the absolute loadings for each factor 
                        will be shown, in addition to the overall (entire model) ECDF of the 
                        absolute loadings.
                        """
                    )
                    cumul_slot = st.empty()

                    factor_loadings = loadings_only[[
                        col for col in loadings_only.columns if col.startswith("factor_")
                    ]]
                    factor_loadings = factor_loadings.abs()
                    factor_loadings_long = factor_loadings.melt(var_name="factor", value_name="value")
                    factor_loadings_copy = factor_loadings_long.copy()
                    factor_loadings_copy["factor"] = "All factors"
                    factor_loadings_long = pd.concat([
                        factor_loadings_long, factor_loadings_copy
                    ], axis=0)
                    del factor_loadings_copy
                    if not show_each_factor:
                        factor_loadings_long = factor_loadings_long[
                            factor_loadings_long["factor"] == "All factors"
                            ]

                    thresh = st.slider(
                        "Absolute threshold", min_value=0.0, max_value=1.0, value=0.35,
                        help="""
                        This discretizes the loadings such that a manifest variable is "included" in the 
                        factor (and its interpretation) if and only if the absolute value of the loading 
                        is at least the threshold. If the value is 1 or -1, then the variable is "included". Otherwise, 
                        it is not.
                        """, key=f"{model_name}_thresh_slider"
                    )
                    threshs.append(thresh)

                    col_1, col_2 = st.columns(2)
                    with col_1:
                        show_raw = st.checkbox("Show original loadings instead?",
                                               key=f"{model_name}_show_raw_instead",
                                               help="""
                                               Leaving this unchecked will show the **binarized** loadings, where a 
                                               manifest variable is either "included" or "not included" in a factor 
                                               based on whether the absolute value of the loading exceeds or equals the 
                                               threshold, or not.
                                               """)
                    with col_2:
                        sort_by_variable = st.checkbox(
                            "Sort by variable name instead?", key=f"{model_name}_sort_by_var",
                            help="By default, the manifest variables are sorted in descending order in terms of "
                                 "their largest loadings. Clicking this will sort them by their name instead."
                        )

                    st.space()

                    fig_cumul = px.ecdf(
                        factor_loadings_long,
                        x="value",
                        color="factor",
                        color_discrete_sequence=discrete_colors,
                        title="Empirical Cumulative Distribution of Absolute Loadings",
                        lines=True,
                        marginal="box"
                    )
                    fig_cumul.update_layout(
                        xaxis_title="Absolute Loading",
                        yaxis_title="Cumulative Probability",
                        legend_title_text="Factor",
                        yaxis=dict(
                            visible=True,
                            showticklabels=True,
                            showline=show_y_line,
                            showgrid=show_y_grid,
                            zeroline=False
                        ),
                        xaxis=dict(
                            visible=True,
                            showticklabels=True,
                            showline=show_x_line,
                            showgrid=show_x_grid,
                            zeroline=False
                        )
                    )
                    fig_cumul.add_vline(
                        x=thresh,
                        line_width=2,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Threshold",
                        annotation_position="top left",
                        row=1,
                        col=1
                    )
                    fig_cumul.add_vline(
                        x=thresh,
                        line_width=2,
                        line_dash="dash",
                        line_color="gray",
                        row=2,
                        col=1
                    )

                    cumul_slot.plotly_chart(fig_cumul, width="stretch", key=f"{model_name}_cumul")

                    df_loadings = loadings_only.copy()
                    if sort_by_variable:
                        df_loadings.sort_index(inplace=True)
                    df_loadings = df_loadings.set_index("variable")
                    factor_cols = [col for col in df_loadings.columns if col.startswith("factor_")]

                    df_loadings_discretized = df_loadings.copy()
                    df_loadings_discretized[factor_cols] = (
                            np.sign(df_loadings_discretized[factor_cols]) *
                            (df_loadings_discretized[factor_cols].abs() >= thresh)
                    ).astype(int)

                    if not show_raw:
                        df_loadings = df_loadings_discretized

                    fig_loadings = px.imshow(
                        df_loadings,
                        text_auto="0.3f" if show_raw else "0",
                        aspect="auto",
                        color_continuous_scale=(continuous_diverging_color_palette
                                                if show_raw else [(0, continuous_diverging_colors[1]),
                                                                  (0.5, "#FFFFFF"),
                                                                  (1, continuous_diverging_colors[-2])]),
                        color_continuous_midpoint=0 if show_raw else 0.5,
                        labels=dict(x="Factors", y="Variables",
                                    color="Loading" if show_raw else "Included"),
                        title="Factor Loading Matrix",
                        zmin=-1,
                        zmax=1
                    )
                    fig_loadings.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                    fig_loadings.update_yaxes(tickmode="linear", dtick=1)
                    fig_loadings.update_layout(
                        height=min(900, max(50 * len(st.session_state.FIT_DETAILS[model_name]["manifest_vars"]), 300))
                    )
                    if not show_raw:
                        fig_loadings.update_coloraxes(showscale=True,
                                                      colorbar_tickvals=[-1, 0, 1],
                                                      colorbar_ticktext=["Yes (neg. loading)",
                                                                         "No",
                                                                         "Yes (pos. loading)"])
                    st.plotly_chart(fig_loadings, width="stretch", key=f"{model_name}_loadings")

                    df_loadings_download = df_loadings.reset_index(
                        inplace=False,
                        drop=False
                    )
                    df_loadings_download.columns = [
                        col.upper() for col in df_loadings_download.columns
                    ]
                    if st.session_state.STATEMENTS_DF is not None:
                        df_loadings_download = pd.merge(
                            left=df_loadings_download, right=st.session_state.STATEMENTS_DF,
                            left_on="VARIABLE", right_on="Variable", how="inner"
                        )
                        df_loadings_download.drop(columns=["Variable"], inplace=True)
                        df_loadings_download.columns = [
                            col.upper() for col in df_loadings_download.columns
                        ]
                        df_loadings_download = df_loadings_download[
                            ["VARIABLE", "STATEMENT"] + [
                                col for col in df_loadings_download.columns
                                if col.startswith("FACTOR_")
                            ]
                            ]
                    st.download_button("Download as CSV file",
                                       df_loadings_download.to_csv(),
                                       key=f"{model_name}_download_loadings",
                                       file_name=f"{model_name}_factor_loadings.csv", width="stretch")
                    st.space()
                    with st.expander("View variables / statements associated with each factor"):
                        for idx, factor in enumerate(df_loadings_discretized.columns):
                            factor_loadings = df_loadings_discretized[factor]
                            variables = df_loadings_discretized[factor_loadings.abs() == 1].index.tolist()
                            variables = sorted(variables, key=lambda x: int(x[1:]))
                            st.markdown(f"#### {factor}")
                            st.write(f"{len(variables)} manifest variable{'s' if len(variables) != 1 else ''}")
                            if len(variables) == 0:
                                st.write("No variables associated.")
                            else:
                                for variable in variables:
                                    if st.session_state.STATEMENTS_DF is not None:
                                        statement = st.session_state.STATEMENTS_DF[
                                            st.session_state.STATEMENTS_DF["Variable"] == variable
                                            ]["Statement"].item()
                                        if statement is None:
                                            st.write(variable)
                                        else:
                                            st.write(f"{variable} - {statement}")
                                    else:
                                        st.write(variable)

                    st.space()

            # Loadings comparison
            cols = st.columns(len(selected_models), border=True)
            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]
                    model_analysis = model_analyses[i]
                    loadings_only = model_analysis[["variable"] + [col for col in model_analysis.columns
                                                                   if col.startswith("factor_")]]
                    st.subheader(":material/compare: Loadings comparison")
                    st.space()

                    loadings_only_long = pd.melt(
                        loadings_only,
                        id_vars="variable",
                        value_vars=[col for col in loadings_only.columns if col.startswith("factor_")],
                        var_name="factor",
                        value_name="loading"
                    )
                    loadings_only_long["x_num"] = loadings_only_long["variable"].str.replace("X", "").astype(int)
                    loadings_only_long["f_num"] = loadings_only_long["factor"].str.replace("factor_", "").astype(int)
                    loadings_only_long = loadings_only_long.sort_values(["x_num", "f_num"]).reset_index(drop=True)

                    sorted_variables = loadings_only_long["variable"].unique().tolist()
                    sorted_factors = loadings_only_long["factor"].unique().tolist()

                    can_proceed = True

                    col_1, col_2 = st.columns(2)
                    with col_1:
                        compare_across = st.selectbox(
                            "Compare across",
                            options=["Manifest Variables", "Latent Factors"],
                            key=f"{model_name}_compare_across"
                        )
                        x_var = "variable" if compare_across == "Latent Factors" else "factor"
                        color_var = "variable" if x_var == "factor" else "factor"
                    with col_2:
                        compare_layout = st.selectbox(
                            "Layout",
                            options=["Combined", "Faceted"],
                            key=f"{model_name}_compare_layout"
                        )

                    if compare_layout == "Faceted":
                        choices = sorted_variables if color_var == "variable" else sorted_factors
                        choices = choices.copy()
                        facet_filter = st.multiselect(
                            "Filter",
                            options=choices,
                            key=f"{model_name}_facet_filter"
                        )

                        if len(facet_filter) > 0:
                            loadings_only_long = loadings_only_long[
                                loadings_only_long[color_var].isin(facet_filter)
                            ]
                            sorted_variables = loadings_only_long["variable"].unique().tolist()
                            sorted_factors = loadings_only_long["factor"].unique().tolist()
                        else:
                            st.warning(f"""
                            Please select at least one {color_var}.
                            """)
                            can_proceed = False

                    if not can_proceed:
                        pass
                    elif compare_layout == "Combined":
                        fig_pseudo_parallel = px.line(
                            loadings_only_long,
                            x=x_var,
                            y="loading",
                            color=color_var,
                            color_discrete_sequence=discrete_colors,
                            category_orders={
                                "variable": sorted_variables,
                                "factor": sorted_factors
                            },
                            title=f"Loadings Across {compare_across}",
                            markers=True
                        )

                        fig_pseudo_parallel.update_traces(line={
                            "width": 3
                        }, opacity=0.8)

                        fig_pseudo_parallel.update_xaxes(
                            type="category",
                            categoryorder="array",
                            categoryarray=(
                                sorted_factors if x_var == "factor" else sorted_variables
                            )
                        )
                        fig_pseudo_parallel.update_xaxes(
                            title_text="Manifest Variable" if x_var == "variable" else "Latent Factor",
                            row=1, col=1
                        )

                        fig_pseudo_parallel.update_yaxes(
                            title_text="Loading",
                        )

                        fig_pseudo_parallel.update_layout(
                            legend_title=color_var.capitalize(),
                            yaxis=dict(
                                visible=True,
                                showticklabels=True,
                                showline=show_y_line,
                                showgrid=show_y_grid,
                                zeroline=False
                            ),
                            xaxis=dict(
                                visible=True,
                                showticklabels=True,
                                showline=show_x_line,
                                showgrid=show_x_grid,
                                zeroline=False
                            )
                        )
                    else:
                        if len(sorted_factors if color_var == "factor" else sorted_variables) == 1:
                            row_spacing = 0.0
                        else:
                            row_spacing = 0.20 / (len(sorted_factors if color_var == "factor"
                                                      else sorted_variables) - 1)

                        fig_pseudo_parallel = px.line(
                            loadings_only_long,
                            x=x_var,
                            y="loading",
                            facet_row=color_var,
                            height=max(350, 200 * len(sorted_factors if color_var == "factor" else sorted_variables)),
                            color=color_var,
                            color_discrete_sequence=discrete_colors,
                            category_orders={
                                "variable": sorted_variables,
                                "factor": sorted_factors
                            },
                            facet_row_spacing=row_spacing,
                            markers=True
                        )

                        fig_pseudo_parallel.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

                        fig_pseudo_parallel.update_traces(line=dict(width=3), opacity=0.8)
                        fig_pseudo_parallel.update_xaxes(
                            type="category",
                            categoryorder="array",
                            categoryarray=(
                                sorted_factors if x_var == "factor" else sorted_variables
                            )
                        )
                        fig_pseudo_parallel.update_xaxes(
                            title_text="Manifest Variable" if x_var == "variable" else "Latent Factor",
                            row=1, col=1,
                            side="bottom",
                            visible=True,
                            showticklabels=True,
                            showline=show_x_line,
                            showgrid=show_x_grid,
                            zeroline=False
                        )
                        fig_pseudo_parallel.update_xaxes(
                            title_text="Manifest Variable" if x_var == "variable" else "Latent Factor",
                            row=len(sorted_factors if color_var == "factor" else sorted_variables),
                            col=1, side="top",
                            visible=True,
                            showticklabels=True,
                            showline=show_x_line,
                            showgrid=show_x_grid,
                            zeroline=False
                        )

                        fig_pseudo_parallel.update_yaxes(
                            title_text="Loading",
                            visible=True,
                            showticklabels=True,
                            showline=show_y_line,
                            showgrid=show_y_grid,
                            zeroline=False
                        )

                        fig_pseudo_parallel.update_layout(
                            showlegend=False,
                            margin=dict(t=100, b=80),
                            title=dict(
                                text=f"Loadings Across {compare_across}",
                                xref="container",
                                yref="container",
                                y=1,
                                x=0,
                                xanchor="left",
                                yanchor="top"
                            )
                        )
                        st.space()

                    if can_proceed:
                        st.plotly_chart(fig_pseudo_parallel, width="stretch", key=f"{model_name}_pseudo_parallel")

                    st.space()

            # Factor cross-loadings
            cols = st.columns(len(selected_models), border=True)
            for i in range(len(selected_models)):
                with cols[i]:
                    sub_col_1, sub_col_2 = st.columns([9, 1])
                    with sub_col_1:
                        st.subheader(":material/network_node: Factor cross-loadings")
                    with sub_col_2:
                        with st.popover("", type="tertiary", icon=":material/help:",
                                        key=f"{model_name}_popover", width="stretch"):
                            st.markdown("""
                            You can interact with the network graph:
                            * Click on **Options** to see the legend and filters.
                            * Hover on the node to see the associated statement, if any.
                            * Click on a node or an edge to move the graph.
                            * Click on a blank space to pan the canvas. Scroll to zoom in or out.
                            * Right-click on the canvas to save the graph as an image.

                            If you cannot see the network graph, try zooming out or try resetting the graph by 
                            unselecting then selecting the model again.
                            """)
                    st.space()

                    st.markdown("""
                    Two nodes are connected if and only if they load high on a common factor, as defined by the
                    absolute threshold in :blue-badge[Factor loadings].
                    """)

                    model_name = selected_models[i]
                    model_analysis = model_analyses[i]
                    loadings_only = model_analysis[["variable"] + [col for col in model_analysis.columns
                                                                   if col.startswith("factor_")]]

                    df_loadings = loadings_only
                    df_loadings = df_loadings.set_index("variable")
                    factor_cols = [col for col in df_loadings.columns if col.startswith("factor_")]

                    df_loadings_discretized = df_loadings
                    df_loadings_discretized[factor_cols] = (
                            np.sign(df_loadings_discretized[factor_cols]) *
                            (df_loadings_discretized[factor_cols].abs() >= threshs[i])
                    ).astype(int)

                    with st.expander("Options"):
                        sub_cols = st.columns(3)
                        for j in range(df_loadings_discretized.shape[1]):
                            with sub_cols[j % 3]:
                                draw_colored_square(df_loadings_discretized.columns[j],
                                                    px.colors.qualitative.Plotly[j])

                        st.caption("""
                        If a pair of manifest variables has more than one factor in common, the 
                        color of the edge connecting the pair will be the average of the colors of the 
                        common factors. Thicker edges indicate more factors in common.
                        """)

                        selected_factors = st.multiselect(
                            "Choose factors to display",
                            options=df_loadings_discretized.columns.tolist(),
                            default=df_loadings_discretized.columns.tolist(),
                            key=f"{model_name}_factors_to_display"
                        )

                        if st.session_state.STATEMENTS is not None:
                            mv_choices = [f"X{i + 1} - {st.session_state.STATEMENTS[i]}"
                                          for i in range(st.session_state.DATA.shape[1])]
                        else:
                            mv_choices = [f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])]
                        selected_mvs = st.multiselect(
                            "Choose manifest variables to display",
                            options=mv_choices,
                            default=mv_choices,
                            key=f"{model_name}_mvs_to_display"
                        )
                        selected_mvs = [selected_mv.split(" - ")[0] for selected_mv in selected_mvs]

                    nodes = []
                    node_font = {
                        "color": "black",
                        "strokeWidth": 3,
                        "strokeColor": "rgba(255, 255, 255, 0.7)"
                    }
                    for mv in st.session_state.FIT_DETAILS[model_name]["manifest_vars"]:
                        if mv not in selected_mvs:
                            continue
                        if st.session_state is not None:
                            title = st.session_state.STATEMENTS_DF[
                                st.session_state.STATEMENTS_DF["Variable"] == mv
                                ]["Statement"].item()
                        else:
                            title = mv
                        nodes.append(
                            Node(
                                id=mv, label=mv, size=10,
                                color="#CDCDCD", font=node_font,
                                title=title
                            )
                        )

                    edge_registry = {}
                    groupings = {
                        col: df_loadings_discretized.index[df_loadings_discretized[col].abs() == 1].tolist()
                        for col in df_loadings_discretized.columns
                    }
                    for factor, manifest_vars in groupings.items():
                        if factor not in selected_factors:
                            continue
                        if len(manifest_vars) == 0:
                            continue
                        factor = int(factor.replace("factor_", "")) - 1
                        for node_1, node_2 in combinations(manifest_vars, 2):
                            if node_1 not in selected_mvs or node_2 not in selected_mvs:
                                continue
                            edge_id = tuple(sorted([node_1, node_2]))
                            if edge_id in edge_registry.keys():
                                edge_registry[edge_id].append(factor)
                            else:
                                edge_registry[edge_id] = [factor]

                    edges = []
                    for edge_id, factors in edge_registry.items():
                        colors = [
                            rgb_to_hex(discrete_colors[factor])
                            if "rgb" in discrete_colors[factor] else discrete_colors[factor]
                            for factor in factors
                        ]
                        final_color = get_mean_color(colors)
                        edges.append(Edge(source=edge_id[0], target=edge_id[1], color=hex_to_rgba(final_color, 0.5),
                                          width=(2 + 2 * len(colors))))

                    network_config = Config(
                        width=None,
                        height=500,
                        directed=False,
                        physics=True,
                        nodeHighlightBehavior=True,
                        updateDelay=100,
                        panAndZoom=True,
                        staticGraph=False
                    )

                    if len(nodes) > 0:
                        selected_node = agraph(nodes, edges, network_config)
                    else:
                        st.warning("Select at least one manifest variable.")

                    st.space()

            # Interpretation
            cols = st.columns(len(selected_models), border=True)
            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]
                    model_analysis = model_analyses[i]
                    loadings_only = model_analysis[["variable"] + [col for col in model_analysis.columns
                                                                   if col.startswith("factor_")]]

                    df_loadings = loadings_only
                    df_loadings = df_loadings.set_index("variable")
                    factor_cols = [col for col in df_loadings.columns if col.startswith("factor_")]

                    df_loadings_discretized = df_loadings
                    df_loadings_discretized[factor_cols] = (
                            np.sign(df_loadings_discretized[factor_cols]) *
                            (df_loadings_discretized[factor_cols].abs() >= threshs[i])
                    ).astype(int)

                    st.subheader(":material/cognition_2: Interpretation")
                    st.space()

                    if st.session_state.STATEMENTS is None:
                        st.warning("The associated statements for the variables were not provided.")
                    else:
                        st.markdown("Interpretations are generated using the groupings defined by the "
                                    "absolute threshold in :blue-badge[Factor loadings].")
                        st.markdown("Note that this is not intended to replace the researcher's judgment and is "
                                    "only meant to help it. For instance, one should cross-check the LLM findings "
                                    "with the sign of the loadings, the cross-loadings, and so on.")
                        st.markdown("""
                        Statements in the "Statement" section follow this format: 
                        **<Variable Name> (<Loading Sign>): <Statement>**. For example, "X1 (Positive): I am sad." 
                        refers to the manifest variable X1, whose statement is "I am sad." and whose loading for the 
                        given factor is positive.
                        """)
                        st.space("xxsmall")

                        has_attempt = st.session_state.INTERPRETATIONS[model_name][0] is not None
                        has_interpretation = st.session_state.INTERPRETATIONS[model_name][0] != "Error"
                        has_successful_interpretation = has_attempt and has_interpretation

                        col_1, col_2, col_3 = st.columns(3)
                        with col_1:
                            st.button("Generate" + (" again" if has_attempt else ""), type="primary",
                                      key=f"{model_name}_interpret_factor_model", width="stretch",
                                      on_click=interpret_factor_model, args=(df_loadings_discretized, model_name))
                        with col_2:
                            st.download_button("Download", type="secondary",
                                               disabled=(not has_successful_interpretation),
                                               data=st.session_state.INTERPRETATIONS[model_name][1],
                                               file_name=f"{model_name}_interpretation.txt",
                                               key=f"{model_name}_download_interpretation", width="stretch")
                        with col_3:
                            st.button("Clear", disabled=(not has_successful_interpretation),
                                      key=f"{model_name}_clear_interpretation", width="stretch",
                                      on_click=interpret_factor_model, args=(None, model_name))

                        st.space()
                        if has_successful_interpretation:
                            st.write(f"Generated by {st.session_state.INTERPRETATIONS[model_name][0]}")
                            st.write(st.session_state.INTERPRETATIONS[model_name][1])
                        elif has_attempt:
                            st.error("Failed to generate interpretation earlier. Please try again.")
                        else:
                            st.warning("Please generate an interpretation first.")

                    st.space()


def about_page():
    st.session_state.CURRENT_PAGE = "about"

    st.session_state.SCROLL_COUNTER = 1 - st.session_state.SCROLL_COUNTER
    with st.container(key=f"app_title_{st.session_state.SCROLL_COUNTER}"):
        st.write("")

    st.markdown("##### :material/person_edit: Project Contributors")
    col_1, col_2, col_3 = st.columns(3, border=True)
    with col_1:
        avatar(
            "./images/avatars/jstuazon_final.jpeg",
            label="Justin Philip Tuazon",
            height=64,
            caption="Developer",
            key="avatar_developer"
        )
        st.markdown("""
        [:material/mail: jstuazon@up.edu.ph](jstuazon@up.edu.ph)
        • [:material/link_2: LinkedIn](https://www.linkedin.com/in/justin-philip-tuazon/)
        """)
        st.markdown("""
        Justin works as a Data Scientist at a large universal bank. He is also an MS Computer Science student at 
        University of the Philippines - Diliman and graduated summa cum laude from the same university with a BS 
        Statistics degree.

        Currently, he is affiliated with the Computer Vision and Machine Intelligence Group of the Department of 
        Computer Science at his university.
        """)
    with col_2:
        avatar(
            "./images/avatars/jeolea_final.jpg",
            label="Joemari Olea",
            height=64,
            caption="Adviser",
            key="avatar_adviser_1"
        )
        st.markdown("""
        [:material/mail: jeolea1@up.edu.ph](jeolea1@up.edu.ph)
        """)
        st.markdown("""
        Joemari is an Assistant Professor at the School of Statistics in the University of the Philippines - 
        Diliman. He holds both MS and BS degrees in Statistics from the same university.

        Currently, he is a doctoral student studying Educational Psychology (Quantitative Methods), specializing in 
        Psychometrics, at the University of Texas at Austin.
        """)
    with col_3:
        avatar(
            "./images/avatars/rbjuayong_final.jpg",
            label="Richelle Ann Juayong",
            height=64,
            caption="Adviser",
            key="avatar_adviser_2"
        )
        st.markdown("""
        [:material/mail: rbjuayong@up.edu.ph](rbjuayong@up.edu.ph)
        """)
        st.markdown("""
        Rich is an Associate Professor at the Department of Computer Science in the University of the Philippines - 
        Diliman. She holds BS, MS. and PhD degrees in Computer Science from the same university.
        
        Currently, she is also a member of the Service Science and Software Engineering Laboratory of the Department 
        of Computer Science at her university. 
        """)

    st.space()

    st.markdown("##### :material/deployed_code: Release and Source")
    col_1, col_2, col_3, col_4 = st.columns(4, border=True)
    with col_1:
        st.caption("Research Paper")
        st.write("Coming soon...")
    with col_2:
        st.caption("Version Number")
        st.write(VERSION_NUMBER)
    with col_3:
        st.caption("Repository")
        st.markdown("[:material/folder_code: GitHub](https://github.com/jptuazon/factorflow)")
    with col_4:
        st.caption("License")
        st.markdown("[:material/license: GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)")

    st.space()


def privacy_page():
    st.session_state.CURRENT_PAGE = "privacy"

    st.session_state.SCROLL_COUNTER = 1 - st.session_state.SCROLL_COUNTER
    with st.container(key=f"app_title_{st.session_state.SCROLL_COUNTER}"):
        st.write("")

    st.markdown("""
    ##### :material/upload: File Uploads
    FactorFlow does not store user uploads in persistent data storage. Uploaded information are discarded when the 
    session ends. The app deals with file uploads using Streamlit's `file_uploader`. Thus, files uploaded to the app 
    are stored only in memory and not in disk. Uploads are deleted when:
    * The user replaced the old file by uploading a new one.
    * The file uploader is cleared.
    * The browser tab where the file was uploaded is closed.

    This means that uploaded files are deleted immediately after they are not in use anymore. More information can 
    be found [here](https://docs.streamlit.io/knowledge-base/using-streamlit/where-file-uploader-store-when-deleted).

    ##### :material/api: Large language Model Calls
    For LLM-related features, FactorFlow uses [Groq](https://groq.com/). Whenever the user generates an 
    interpretation, the app uses Groq's API and passes data about the loading matrix and the statements (but not 
    the raw data). Thus, for LLM calls, the app follows Groq's privacy policy, which you can find 
    [here](https://console.groq.com/docs/legal).

    An example of what is passed to Groq's API is shown below.
    ```
    Scale Direction - Disagree-Agree
    
    factor_1:
    - X1 (Negative): I prefer not to show a partner how I feel deep down.
    - X9 (Negative): I don't feel comfortable opening up to romantic partners.
    - X15 (Positive): I feel comfortable sharing my private thoughts and feelings with my partner.
    - X19 (Positive): I find it relatively easy to get close to my partner.
    - X25 (Positive): I tell my partner just about everything.
    - X27 (Positive): I usually discuss my problems and concerns with my partner.
    - X29 (Positive): I feel comfortable depending on romantic partners.
    - X31 (Positive): I don't mind asking romantic partners for comfort, advice, or help.
    - X32 (Positive): I get frustrated if romantic partners are not available when I need them.
    - X33 (Positive): It helps to turn to my romantic partner in times of need.
    - X35 (Positive): I turn to my partner for many things, including comfort and reassurance.

    factor_2:
    - X2 (Positive): I worry about being abandoned.
    - X4 (Positive): I worry a lot about my relationships.
    - X6 (Positive): I worry that romantic partners wont care about me as much as I care about them.
    - X8 (Positive): I worry a fair amount about losing my partner.
    - X10 (Positive): I often wish that my partner's feelings for me were as strong as my feelings for him/her.
    - X14 (Positive): I worry about being alone.
    - X18 (Positive): I need a lot of reassurance that I am loved by my partner.
    - X22 (Negative): I do not often worry about being abandoned.
    - X28 (Positive): When I'm not involved in a relationship, I feel somewhat anxious and insecure.
    - X34 (Positive): When romantic partners disapprove of me, I feel really bad about myself.

    factor_3:
    - X5 (Positive): Just when my partner starts to get close to me I find myself pulling away.
    - X7 (Positive): I get uncomfortable when a romantic partner wants to be very close.
    - X13 (Positive): I am nervous when partners get too close to me.
    - X30 (Negative): I get frustrated when my partner is not around as much as I would like.
    - X32 (Negative): I get frustrated if romantic partners are not available when I need them.
    - X36 (Negative): I resent it when my partner spends time away from me.

    factor_4:
    - X3 (Negative): I am very comfortable being close to romantic partners.
    - X5 (Positive): Just when my partner starts to get close to me I find myself pulling away.
    - X6 (Positive): I worry that romantic partners wont care about me as much as I care about them.
    - X7 (Positive): I get uncomfortable when a romantic partner wants to be very close.
    - X9 (Positive): I don't feel comfortable opening up to romantic partners.
    - X10 (Positive): I often wish that my partner's feelings for me were as strong as my feelings for him/her.
    - X11 (Positive): I want to get close to my partner, but I keep pulling back.
    - X13 (Positive): I am nervous when partners get too close to me.
    - X17 (Positive): I try to avoid getting too close to my partner.
    - X18 (Positive): I need a lot of reassurance that I am loved by my partner.
    - X19 (Negative): I find it relatively easy to get close to my partner.
    - X20 (Positive): Sometimes I feel that I force my partners to show more feeling, more commitment.
    - X21 (Positive): I find it difficult to allow myself to depend on romantic partners.
    - X23 (Positive): I prefer not to be too close to romantic partners.
    - X24 (Positive): If I can't get my partner to show interest in me, I get upset or angry.
    - X26 (Positive): I find that my partner(s) don't want to get as close as I would like.
    - X30 (Positive): I get frustrated when my partner is not around as much as I would like.
    - X32 (Positive): I get frustrated if romantic partners are not available when I need them.
    - X36 (Positive): I resent it when my partner spends time away from me.

    factor_5:
    - X1 (Negative): I prefer not to show a partner how I feel deep down.
    - X3 (Positive): I am very comfortable being close to romantic partners.
    - X7 (Negative): I get uncomfortable when a romantic partner wants to be very close.
    - X9 (Negative): I don't feel comfortable opening up to romantic partners.
    - X10 (Positive): I often wish that my partner's feelings for me were as strong as my feelings for him/her.
    - X12 (Positive): I often want to merge completely with romantic partners, and this sometimes scares them away.
    - X15 (Positive): I feel comfortable sharing my private thoughts and feelings with my partner.
    - X16 (Positive): My desire to be very close sometimes scares people away.
    - X20 (Positive): Sometimes I feel that I force my partners to show more feeling, more commitment.
    - X23 (Negative): I prefer not to be too close to romantic partners.
    - X25 (Positive): I tell my partner just about everything.
    - X26 (Positive): I find that my partner(s) don't want to get as close as I would like.
    - X27 (Positive): I usually discuss my problems and concerns with my partner.
    - X35 (Positive): I turn to my partner for many things, including comfort and reassurance.
    ```
    """)


pages = [
    st.Page(overview_page, title=":material/home: Overview"),
    st.Page(diagnostics_page, title=":material/data_thresholding: Diagnostics"),
    st.Page(dashboard_page, title=":material/dashboard: Dashboard"),
    st.Page(about_page, title=":material/page_info: About"),
    st.Page(privacy_page, title=":material/privacy_tip: Privacy")
]
pg = st.navigation(pages, position="top")
pg.run()

if st.session_state.CURRENT_PAGE != "overview":
    st.logo("./images/factor_flow_logo.png", size="large",
            link="https://github.com/jptuazon/factorflow")

if show_floating_top:
    if floating_button(":material/keyboard_double_arrow_up: Top"):
        scroll_to_element(f"app_title_{st.session_state.SCROLL_COUNTER}")

footer = """
<div class="custom-footer">
    FactorFlow: A Visual Analytics Workspace with Large Language Model–Assisted Interpretation for Factor Analysis
    • Copyright © 2026 Justin Philip Tuazon
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
