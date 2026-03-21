# Copyright 2026 Justin Philip Tuazon

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

# FactorFlow V1.0.2
# https://factorflow-efa.streamlit.app/

import json
import math
import time
from itertools import product, combinations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from interpretablefa import InterpretableFA
from streamlit_js_eval import streamlit_js_eval


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
        if (!window.use_model) {{
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
if "DATA" not in st.session_state:
    st.session_state.DATA = None

if "DATA_NAME" not in st.session_state:
    st.session_state.DATA_NAME = None

if "STATEMENTS" not in st.session_state:
    st.session_state.STATEMENTS = None
if "STATEMENTS_DF" not in st.session_state:
    st.session_state.STATEMENTS_DF = None

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

if "FACTOR_MODELS" not in st.session_state:
    st.session_state.FACTOR_MODELS = None
if st.session_state.DATA is not None and st.session_state.FACTOR_MODELS is None:
    st.session_state.FACTOR_MODELS = InterpretableFA(st.session_state.DATA)

if "FIT_DETAILS" not in st.session_state:
    st.session_state.FIT_DETAILS = {}

if "FIT_MODEL" not in st.session_state:
    st.session_state.FIT_MODEL = "No"

if "model_name" not in st.session_state:
    st.session_state.model_name = None

if "number_of_factors" not in st.session_state:
    st.session_state.number_of_factors = None

if "rotation" not in st.session_state:
    st.session_state.rotation = None

if "prior_matrix" not in st.session_state:
    st.session_state.prior_matrix = None

if "prior" not in st.session_state:
    st.session_state.prior = None

if "SAMPLE_V_DATA" not in st.session_state:
    st.session_state.SAMPLE_V_DATA = pd.read_csv("./sample_data/sample_v_data.csv")

if "CURRENT_LLM_MODEL_ID" not in st.session_state:
    st.session_state.CURRENT_LLM_MODEL_ID = None

if "INTERPRETATIONS" not in st.session_state:
    st.session_state.INTERPRETATIONS = {}

if "STATEMENT_TAGS" not in st.session_state:
    st.session_state.STATEMENT_TAGS = {}


# Fit factor model
@st.dialog("Fit a new model", dismissible=False)
def fit_factor_model():
    if st.session_state.FIT_MODEL == "Yes":
        with st.spinner("Fitting the factor model...", width="stretch", show_time=True):
            model_name = str(st.session_state.model_name)
            number_of_factors = int(st.session_state.number_of_factors)
            rotation = None if st.session_state.rotation == "None" else str(st.session_state.rotation).lower()
            prior_matrix = st.session_state.prior_matrix

            st.session_state.FACTOR_MODELS.fit_factor_model(
                model_name,
                number_of_factors,
                rotation,
                prior_matrix
            )

            st.session_state.FIT_DETAILS[model_name] = {
                "number_of_factors": number_of_factors,
                "rotation": rotation,
                "prior": st.session_state.prior
            }

            st.session_state.INTERPRETATIONS[model_name] = (None, "")

        st.session_state.model_name = None
        st.session_state.number_of_factors = None
        st.session_state.rotation = None
        st.session_state.prior_matrix = None
        st.session_state.FIT_MODEL = "No"

    st.success("Factor model created.")
    st.space()
    col_1, col_2 = st.columns([15, 3])
    with col_2:
        if st.button("Finish", width="stretch"):
            st.rerun()


if st.session_state.FIT_MODEL == "Yes" and (st.session_state.FACTOR_MODELS is not None):
    fit_factor_model()

# LLM set up
LLM_API_KEY = st.secrets["GROQ_API_KEY"]
LLM_MODEL_IDS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b"
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
                    You are an expert in exploratory factor analysis. Your job is to be an "EFA Factor 
                    Interpretation Assistant". Basically, for each factor in a factor model, you have to come up 
                    with a short but appropriate factor label that encapsulates the statements associated with it. 
                    You also have to provide a description of the factor. Finally, you need to provide a 
                    justification of your labels and descriptions. Cite specific statements that led you to your 
                    interpretation for each factor.
                    
                    The input will be given to you in the following format:
                    
                        factor_1:
                        - [STATEMENT 1]
                        - [STATEMENT 2]
                        factor_2:
                        - [STATEMENT 3]
                        - [STATEMENT 4]
                        
                    The number of factors will vary and the number of statements for each factor will also vary.
                    
                    Requirements:
                    1. Each label should be 2-4 words.
                    2. Each description should explain the underlying latent construct.
                    3. Each justification should explain why the label and description are appropriate.
                    4. Strictly follow the format of the sample output. Do not add any other filler text such as 
                    "Based on the input, here's what I think.".
                    5. Make sure to do the procedure for every factor.
                    6. Format the output properly. Bold the factor name, add line breaks between the label, the 
                    description, and the justification. Bold "Label", "Description", and "Justification", too. Add a 
                    horizontal line between each factor block, as shown in the sample.
                    7. In the justification, feel free to be creative but logical. In addition to citing specific 
                    statements, provide additional explanation.
                    
                    Sample output:
                    
                    factor_1
                    
                    • Label: Nostalgia
                    
                    • Description: This factor refers to how much nostalgia drives the purchasing behavior 
                    of the consumer.
                    
                    • Justification: The statements "I am loyal to brands I have used before." and "I associate 
                    products with memories." suggest that nostalgia drives this factor.
                    
                    -----------------------------------
                    
                    factor_2
                    
                    • Label: Practicality
                    
                    • Description: This factor refers to how much practicality drives the purchasing behavior of 
                    the consumer.
                    
                    • Justification: The statements "I look for functionality over aesthetics." and "I always 
                    consider my budget when buying a product." suggest that this factor refers to how much 
                    practicality is valued.
                    
                    """
                },
                {
                    "role": "user",
                    "content": factors
                }
            ],
            max_completion_tokens=2048,
            temperature=0.05,
            top_p=0.95,
            stream=False
        )

        return llm_model_id, interpretation.choices[0].message.content.replace("\n", "  \n")
    except Exception:
        return "Error", "Rate limit reached. Please try again in a while or try a different model."


def interpret_factor_model(df_discretized_loadings):
    input_for_llm = ""

    for idx, factor in enumerate(df_discretized_loadings.columns):
        factor_loadings = df_discretized_loadings[factor]
        variables = df_discretized_loadings[factor_loadings == 1].index.tolist()

        if len(variables) == 0:
            continue
        else:
            input_for_llm += f"\n{factor}:\n"

            for variable in variables:
                statement = st.session_state.STATEMENTS_DF[
                    st.session_state.STATEMENTS_DF["Variable"] == variable
                ]["Statement"].item()
                input_for_llm += f"- {statement}\n"

    input_for_llm = input_for_llm.encode("utf-8").decode("unicode_escape")

    return generate_interpretation(input_for_llm)


# Page configuration
st.set_page_config(
    page_title="FactorFlow",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": """
        * Version Number: 1.0.1
        * FactorFlow was developed by Justin Philip Tuazon. You may reach out via email at jstuazon@alum.up.edu.ph or  
        [LinkedIn](https://www.linkedin.com/in/justin-philip-tuazon/).
        * The pairwise target rotation method used here was authored by Justin Philip Tuazon, Gia Mizrane Abubo, and 
        Joemari Olea. The pre-print of the method can be found [here](https://arxiv.org/abs/2409.11525).
        """
    }
)

# App constants
ORTHOGONAL_ROTATIONS = ["Priorimax", "Varimax", "Oblimax", "Quartimax", "Equamax"]
OBLIQUE_ROTATIONS = ["Promax", "Oblimin", "Quartimin"]
ROTATIONS = ORTHOGONAL_ROTATIONS + OBLIQUE_ROTATIONS + ["None"]


# App functions
def compute_tags_breakdown(df_loadings):
    df_loadings = df_loadings.copy().reset_index(drop=False)
    factor_cols = [col for col in df_loadings.columns if col.startswith("factor_")]
    df_loadings[factor_cols] = df_loadings[factor_cols] ** 2

    tag_variable = {}
    for statement, tags in st.session_state.STATEMENT_TAGS.items():
        variable = st.session_state.STATEMENTS_DF[
            st.session_state.STATEMENTS_DF["Statement"] == statement
        ]["Variable"].item()
        for tag in tags:
            if tag not in tag_variable:
                tag_variable[tag] = [variable]
            else:
                tag_variable[tag].append(variable)

    tags = []
    factors = []
    absolute_loadings_sum = []
    for factor, tag in product(factor_cols, list(tag_variable.keys())):
        variables = tag_variable[tag]
        factor_absolute_loadings = df_loadings[["variable", factor]].copy()
        factor_absolute_loadings["included"] = factor_absolute_loadings["variable"].isin(variables).astype(int)
        total = factor_absolute_loadings[
            factor_absolute_loadings["included"] == 1
        ][factor].sum()
        tags.append(tag)
        factors.append(factor)
        absolute_loadings_sum.append(total)

        del factor_absolute_loadings

    tagged_variables = set([variable for variables in tag_variable.values() for variable in variables])
    untagged_variables = [f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])
                          if f"X{idx + 1}" not in tagged_variables]
    for factor in factor_cols:
        variables = untagged_variables
        factor_absolute_loadings = df_loadings[["variable", factor]].copy()
        factor_absolute_loadings["included"] = factor_absolute_loadings["variable"].isin(variables).astype(int)
        total = factor_absolute_loadings[
            factor_absolute_loadings["included"] == 1
        ][factor].sum()
        tags.append("No tag")
        factors.append(factor)
        absolute_loadings_sum.append(total)

        del factor_absolute_loadings

    df_tags_breakdown = pd.DataFrame({
        "Factor": factors,
        "Tag": tags,
        "Sum of Squared Loadings": absolute_loadings_sum
    })

    del df_loadings

    return df_tags_breakdown


def process_prior_matrix(prior_matrix, rotation):
    result = {
        "pass": True,
        "message": "Passed."
    }

    if prior_matrix is None:
        if rotation.lower == "priorimax":
            result["pass"] = False
            result["message"] = "Priormax rotation requires a prior matrix."
        return result

    if len(prior_matrix.shape) != 2:
        result["pass"] = False
        result["message"] = "The custom prior matrix must be 2D."
        return result

    if (prior_matrix.shape[0] != st.session_state.DATA.shape[1]
            or prior_matrix.shape[1] != st.session_state.DATA.shape[1]):
        result["pass"] = False
        result["message"] = ("The number of rows (or columns) of the prior matrix must match the number of manifest "
                             "variables (i.e., the number of columns in the main dataset).")
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
                    result["message"] = "All entries must be either a number or blank."
                    break

                if val < 0 or val > 1:
                    result["pass"] = False
                    result["message"] = "All entries must be between 0 and 1 (inclusive)."
                    break

            if val is None:
                if prior_matrix[col, row] is None or prior_matrix[col, row] == "":
                    prior_matrix[col, row] = None
                    continue
                else:
                    result["pass"] = False
                    result["message"] = "The matrix must be symmetric."
                    break
            else:
                if prior_matrix[col, row] is None or prior_matrix[col, row] == "":
                    result["pass"] = False
                    result["message"] = "The matrix must be symmetric."
                    break
                else:
                    if not np.isclose(val, prior_matrix[col, row]):
                        result["pass"] = False
                        result["message"] = "The matrix must be symmetric."
                        break
                    else:
                        prior_matrix[col, row] = val

    return result


@st.dialog("Upload dataset", width="medium")
def upload_data_dialog():
    df_data = None
    data_file_name = None
    statements = None

    csv_data = st.file_uploader("Choose a CSV file for the main dataset:", type="csv")
    if csv_data is not None:
        df_data = pd.read_csv(csv_data, header=None)
        df_data.columns = [F"X{i + 1}" for i in range(len(df_data.columns))]
        df_data = df_data.apply(pd.to_numeric, errors="coerce")
        data_file_name = csv_data.name
        if df_data.isnull().values.any() > 0:
            st.error("The dataset must have only numeric values and there must be no missing values.")
            st.stop()
        else:
            st.success("This dataset is valid.")

    if df_data is not None:
        use_statements = st.checkbox("Upload statements or questions associated with the manifest variables?")
        if use_statements:
            txt_data = st.file_uploader("Choose a TXT file for the statements:", type="txt")
            if txt_data is not None:
                statements = txt_data.getvalue().decode("utf-8")
                statements = statements.splitlines()
                if len(statements) != df_data.shape[1]:
                    st.error("The number of statements must match the number of columns of the main dataset.")
                    st.stop()
                else:
                    st.success("This list of statements is valid.")
            else:
                st.warning("Please upload the CSV file.")
                st.stop()

    st.space()
    if df_data is not None:
        col_1, col_2 = st.columns([15, 3])
        with col_2:
            if st.button("Confirm", width="stretch"):
                st.session_state.DATA = df_data
                st.session_state.DATA_NAME = data_file_name
                st.session_state.STATEMENTS = statements
                st.session_state.STATEMENTS_DF = pd.DataFrame({
                    "Variable": [f"X{idx + 1}" for idx in range(df_data.shape[1])],
                    "Statement": statements
                })
                if statements is not None and len(statements) > 0:
                    for statement in statements:
                        st.session_state.STATEMENT_TAGS[statement] = []
                st.rerun()


@st.dialog("Statement tags", width="large")
def view_tags_dialog():
    st.write("""
    Tags are a priori labellings of statements. For instance, you can tag the statements "I am loyal to brands I have 
    used before." and "I associate products with memories." with "Nostalgia". Each statement can have zero or more 
    tags. These tags are then used to summarize the "breakdown" of a factor in the *Dashboard*.
    """)

    df_tags = pd.DataFrame([
        {"Statement": statement, "Tags": ", ".join(tags)}
        for statement, tags in st.session_state.STATEMENT_TAGS.items()
    ])
    df_tags.index = [f"X{int(idx + 1)}" for idx in df_tags.index]
    df_tags.sort_values(by=["Statement"], ascending=[True], inplace=True)
    st.dataframe(df_tags)

    st.space()
    col_1, col_2 = st.columns([6, 1])
    with col_2:
        if st.button("Clear all tags", width="stretch"):
            st.session_state.STATEMENT_TAGS = {}
            st.rerun()

    st.markdown("## Edit tags")

    input_type = st.selectbox(
        "Input type",
        options=["Manual", "Upload"]
    )
    current_statement = None
    df_new_tags = None

    if input_type == "Manual":
        st.markdown("### Current statement:")
        current_statement = st.selectbox(
            "Choose statement",
            options=sorted(list(st.session_state.STATEMENTS))
        )

        current_tags = ", ".join(list(st.session_state.STATEMENT_TAGS[current_statement]))
        st.markdown("### Current tags:")
        if current_tags != "":
            st.write(current_tags)
        else:
            st.write("No tags yet")

        st.markdown("### New tags:")
        new_tags = st.multiselect(
            "Enter new tags",
            options=list(st.session_state.STATEMENT_TAGS[current_statement]),
            accept_new_options=True,
            placeholder="Selecting no tag will remove all tags.."
        )
    else:
        new_tags = st.file_uploader(label="Upload CSV file for the statement tags", type="csv", width="stretch",
                                    help="""
                                    The CSV file must have two columns. The first column must contain the statements 
                                    and the second column must contain the corresponding tags. If a statement has 
                                    multiple tags, separate them using commas (e.g., "Tag A, Tag B"). If a statement 
                                    has no tags, leave the cell blank. The CSV file must not have headers.
                                    """)
        if new_tags is not None:
            df_new_tags = pd.read_csv(new_tags, header=None)

            if df_new_tags.shape[1] != 2:
                st.error("The CSV file must have two columns, first for the statement and second for the tag.")
                st.stop()
            df_new_tags.columns = ["statement", "tags"]

            if set(df_new_tags["statement"]) != set(st.session_state.STATEMENTS):
                st.error("The set of statements in the CSV file must match exactly the set of statements "
                         "originally uploaded.")
                st.stop()

    st.space()
    col_1, col_2 = st.columns([6, 1])
    with col_2:
        if st.button("Confirm", width="stretch"):
            if input_type == "Manual":
                st.session_state.STATEMENT_TAGS[current_statement] = new_tags
            else:
                st.write(df_new_tags)
                for idx, row in df_new_tags.iterrows():
                    statement = row["statement"]
                    tags = row["tags"].split(",") if not pd.isna(row["tags"]) else []
                    tags = [tag.strip() for tag in tags if tag.strip() != ""] if len(tags) > 0 else tags
                    if "No tag" in tags:
                        st.error(""""No tag" is a reserved tag. Please remove this tag from your file.""")
                        st.stop()
                    st.session_state.STATEMENT_TAGS[statement] = tags
            st.rerun()


@st.dialog("View basic stats", width="large")
def view_data_dialog():
    with st.expander("Raw data"):
        st.dataframe(st.session_state.DATA)

    with st.expander("Summary statistics"):
        st.dataframe(st.session_state.DATA.describe())

    with st.expander("Correlation matrix"):
        corr_mat = st.session_state.DATA.corr()
        fig_corr = px.imshow(
            corr_mat,
            text_auto="0.3f",
            aspect="auto",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        fig_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
        fig_corr.update_yaxes(tickmode="linear", dtick=1)
        fig_corr.update_layout(
            height=20 * st.session_state.DATA.shape[1]
        )
        st.plotly_chart(fig_corr, width="stretch")
        st.dataframe(corr_mat)

    with st.expander("Associated statement for each variable"):
        if st.session_state.STATEMENTS is None:
            st.write("Associated statements are not available / were not uploaded.")
        else:
            var_statement = pd.DataFrame(
                {
                    "Variable": [f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])],
                    "Statement": st.session_state.STATEMENTS,
                    "Mean": [
                        st.session_state.DATA[col].mean() for col in st.session_state.DATA.columns
                    ]
                }
            )
            st.dataframe(var_statement)

    with st.expander("Semantic similarity matrix"):
        if st.session_state.STATEMENTS is None:
            st.write("Associated statements are not available / were not uploaded.")
        else:
            embeddings = st.session_state.EMBEDDINGS
            if embeddings is not None:
                semantic_similarity_mat = st.session_state.SEMANTIC_SIMILARITY_MATRIX
                fig_semantic = px.imshow(
                    semantic_similarity_mat,
                    text_auto="0.3f",
                    aspect="auto",
                    color_continuous_scale='Blues',
                    zmin=0, zmax=1
                )
                fig_semantic.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                fig_semantic.update_yaxes(tickmode="linear", dtick=1)
                fig_semantic.update_layout(
                    height=20 * st.session_state.DATA.shape[1]
                )
                st.plotly_chart(fig_semantic, width="stretch")
                st.dataframe(semantic_similarity_mat)
            else:
                st.write(":hourglass_flowing_sand: Loading...")

    with (st.expander("Sampling Adequacy")):
        st.markdown("#### Kaiser-Meyer-Olkin")
        temp = InterpretableFA(st.session_state.DATA)
        df_kmo = pd.DataFrame({
            "Variable": [f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])] + ["Overall"],
            "KMO": list(temp.kmo[0]) + [temp.kmo[1]]
        })
        st.dataframe(df_kmo)
        st.markdown("#### Test for Sphericity")
        st.write(f"Test statistic: {temp.sphericity[0]}")
        st.write(f"p-value: {temp.sphericity[1]}")


@st.dialog("Fit a new model", width="large", dismissible=False)
def fit_model_dialog():
    model_name = None
    number_of_factors = None
    rotation = None
    prior = None
    prior_matrix = None

    st.markdown("""
    :bulb: For the model name, it is recommended to follow this format: 
    [rotation]\\_[number of factors]\\_[prior matrix description] (e.g., "priorimax_3_semantics", "varimax_5_custom").
    """)

    can_proceed = True

    col_1, col_2 = st.columns(2)
    with col_1:
        model_name = st.text_input(label="Model name", value=f"model_{len(st.session_state.FACTOR_MODELS.models) + 1}",
                                   help="This must be unique.", width="stretch")
        rotation = st.selectbox(label="Rotation", placeholder="Select a rotation method...",
                                options=ROTATIONS, help="Priorimax, Varimax, Oblimax, Quartimax, and Equamax are "
                                                        "orthogonal rotations. Promax, Oblimin, and Quartimin are "
                                                        "oblique rotations.", width="stretch")
    with col_2:
        number_of_factors = st.number_input(label="Number of factors", placeholder="Enter a positive number...",
                                            value=3, min_value=1, max_value=(st.session_state.DATA.shape[1] - 1),
                                            help="The number of factors cannot exceed the number of manifest "
                                                 "variables.", width="stretch")
        prior = st.selectbox(label="Prior matrix", placeholder="Specify the prior matrix for priorimax...",
                             options=("Semantics", "Grouped", "Custom", "None"),
                             help="""
                             "Semantics" uses the semantic similarity matrix. "Custom" lets you specify the exact 
                             matrix by uploading a CSV file. This is required when using the priorimax rotation and 
                             optional for other rotation methods.""", width="stretch")

    if prior == "Semantics":
        if st.session_state.SEMANTIC_SIMILARITY_MATRIX is None:
            st.warning("You cannot use the semantic similarity matrix for the prior matrix since no statements "
                       "were loaded.")
            can_proceed = False
        if can_proceed:
            semantic_similarity_mat = st.session_state.SEMANTIC_SIMILARITY_MATRIX
            fig_semantic = px.imshow(
                semantic_similarity_mat,
                text_auto="0.3f",
                aspect="auto",
                color_continuous_scale='Blues',
                zmin=0, zmax=1
            )
            fig_semantic.update_xaxes(side="bottom", tickmode="linear", dtick=1)
            fig_semantic.update_yaxes(tickmode="linear", dtick=1)
            fig_semantic.update_layout(
                height=20 * st.session_state.DATA.shape[1],
                title="The semantic similarity matrix will be used as the prior matrix."
            )
            st.plotly_chart(fig_semantic, width="stretch")
            prior_matrix = semantic_similarity_mat.to_numpy()
    elif prior == "Grouped":
        cols = st.columns(3)
        if st.session_state.STATEMENTS is not None:
            group_choices = [f"X{i + 1} - {st.session_state.STATEMENTS[i]}"
                             for i in range(st.session_state.DATA.shape[1])]
        else:
            group_choices = [f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])]
        groupings = []
        if number_of_factors is not None:
            for factor_number in range(number_of_factors):
                current_col = cols[factor_number % len(cols)]
                with current_col:
                    groupings.append(
                        st.multiselect(label=f"Grouping {factor_number + 1}", options=group_choices)
                    )

        for idx, grouping in enumerate(groupings):
            if len(grouping) < 2:
                st.warning(f"Grouping {idx + 1} must have at least two entries.")
                can_proceed = False
            groupings[idx] = [int(val.split(" - ")[0].replace("X", "")) - 1 for val in grouping]

        if can_proceed:
            prior_matrix = np.zeros((st.session_state.DATA.shape[1], st.session_state.DATA.shape[1]), dtype=np.int8)
            np.fill_diagonal(prior_matrix, 1)
            for grouping in groupings:
                for pair in combinations(grouping, 2):
                    prior_matrix[pair[0], pair[1]] = 1
                    prior_matrix[pair[1], pair[0]] = 1

            df_prior_matrix = pd.DataFrame(prior_matrix,
                                           index=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])],
                                           columns=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])])
            fig_prior = px.imshow(
                df_prior_matrix,
                text_auto="0",
                aspect="auto",
                color_continuous_scale='Blues',
                zmin=0, zmax=1
            )
            fig_prior.update_xaxes(side="bottom", tickmode="linear", dtick=1)
            fig_prior.update_yaxes(tickmode="linear", dtick=1)
            fig_prior.update_layout(
                height=20 * st.session_state.DATA.shape[1],
                title="This grouping matrix will be used as the prior matrix."
            )
            st.plotly_chart(fig_prior, width="stretch")
    elif prior == "Custom":
        prior_matrix = st.file_uploader(label="Upload CSV file for the prior matrix", type="csv", width="stretch",
                                        disabled=(prior != "Custom"))
        if prior_matrix is not None:
            prior_matrix = pd.read_csv(prior_matrix, header=None).to_numpy()

            check = process_prior_matrix(prior_matrix, rotation)
            if check["pass"]:
                df_prior_matrix = pd.DataFrame(prior_matrix,
                                               index=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])],
                                               columns=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])])
                fig_prior = px.imshow(
                    df_prior_matrix,
                    text_auto="0.3f",
                    aspect="auto",
                    color_continuous_scale='Blues',
                    zmin=0, zmax=1
                )
                fig_prior.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                fig_prior.update_yaxes(tickmode="linear", dtick=1)
                fig_prior.update_layout(
                    height=20 * st.session_state.DATA.shape[1],
                    title="This custom matrix will be used as the prior matrix."
                )
                st.plotly_chart(fig_prior, width="stretch")
            else:
                st.error(check["message"])
                can_proceed = False
        else:
            st.warning("Please upload the CSV file.")
            can_proceed = False
    elif prior == "None":
        prior_matrix = None

    st.space()
    col_1, col_2, col_3 = st.columns([12, 3, 3])
    with col_3:
        if st.button("Cancel", width="stretch"):
            st.rerun()
    if not can_proceed:
        st.stop()
    with col_2:
        if st.button("Fit model", width="stretch"):
            any_failed = False

            if model_name.strip() == "" or model_name is None:
                st.toast("🚫 The model name cannot be empty.", duration="long")
                any_failed = True
            if model_name in st.session_state.FACTOR_MODELS.models.keys():
                st.toast("🚫 The model name must be unique.")
                any_failed = True

            try:
                number_of_factors = int(number_of_factors)
            except ValueError:
                st.toast("🚫 The number of factors must be an integer.", duration="long")
                any_failed = True
            if number_of_factors < 1 or number_of_factors >= st.session_state.DATA.shape[1]:
                st.toast("🚫 The number of factors must be at least 1 but less than the number of "
                         "manifest variables.", duration="long")
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

            check = process_prior_matrix(prior_matrix, rotation)
            if not check["pass"]:
                st.toast("🚫" + check["message"], duration="long")
                any_failed = True

            if any_failed:
                st.stop()

            st.session_state.model_name = model_name
            st.session_state.number_of_factors = number_of_factors
            st.session_state.rotation = rotation
            st.session_state.prior_matrix = prior_matrix
            st.session_state.prior = prior
            st.session_state.FIT_MODEL = "Yes"
            st.rerun()


def delete_factor_model(model_name):
    st.session_state.FACTOR_MODELS.remove_factor_model(model_name)
    del st.session_state.FIT_DETAILS[model_name]
    if model_name in st.session_state.INTERPRETATIONS:
        del st.session_state.INTERPRETATIONS[model_name]


@st.dialog("View factor models", width="large", on_dismiss="rerun")
def view_models_dialog():
    model_name = st.selectbox("Choose a model", options=sorted(list(st.session_state.FACTOR_MODELS.models.keys())),
                              width="stretch")

    if model_name is None:
        st.error("There are no factor models available.")
    else:
        col_1, col_2, col_3 = st.columns([15, 6, 3])
        with col_2:
            if st.button("Clear saved interpretation", width="stretch"):
                st.session_state.INTERPRETATIONS[model_name] = (None, "")
                st.rerun()
        with col_3:
            st.button("Delete model", on_click=delete_factor_model, args=(model_name, ),
                      width="stretch")

        st.subheader("Fit details")

        factor_model = st.session_state.FACTOR_MODELS.models[model_name]
        fit_details = st.session_state.FIT_DETAILS[model_name]

        col_1, col_2, col_3, col_4 = st.columns([3, 3, 3, 3])
        with col_1:
            st.markdown("#### Number of factors")
            st.badge(str(fit_details["number_of_factors"]), color="green")
        with col_2:
            st.markdown("#### Rotation")
            st.badge(str(fit_details["rotation"]).capitalize(), color="green")
        with col_3:
            st.markdown("#### Prior type")
            st.badge(str(fit_details["prior"]), color="green")
        with col_4:
            st.markdown("#### V index")
            v = st.session_state.FACTOR_MODELS.calculate_v_index(model_name)
            v = np.round(v, 5) if v is not None else None
            st.badge(str(v), color="green")

        if fit_details["prior"] != "None":
            with st.expander("View prior matrix", expanded=False):
                df_prior_matrix = pd.DataFrame(factor_model.prior_,
                                               index=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])],
                                               columns=[f"X{i + 1}" for i in range(st.session_state.DATA.shape[1])])
                fig_prior = px.imshow(
                    df_prior_matrix,
                    text_auto="0.3f",
                    aspect="auto",
                    color_continuous_scale='Blues',
                    zmin=0, zmax=1
                )
                fig_prior.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                fig_prior.update_yaxes(tickmode="linear", dtick=1)
                fig_prior.update_layout(
                    height=20 * st.session_state.DATA.shape[1],
                    title="Prior matrix"
                )
                st.plotly_chart(fig_prior, width="stretch")
                col_1, col_2, col_3 = st.columns(3)
                with col_2:
                    st.download_button("Download prior matrix as CSV file", df_prior_matrix.to_csv(),
                                       file_name="prior_matrix.csv", width="stretch")
                st.space()
        else:
            st.warning("No prior matrix was used for this model.")

        st.subheader("Model details")
        model_analysis = st.session_state.FACTOR_MODELS.analyze_model(model_name).reset_index(drop=True)
        means = pd.DataFrame({
            "variable": [f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])],
            "mean": [st.session_state.DATA[f"X{idx + 1}"].mean() for idx in range(st.session_state.DATA.shape[1])]
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
            cmap="RdBu", axis=None, subset=[col for col in model_analysis.columns if col.startswith("factor_")],
            vmin=-1.0, vmax=1.0
        ).background_gradient(
            cmap="Purples", axis=0, subset=["COMMUNALITY", "KMO_MSA"], vmin=0, vmax=1.0
        )

        st.markdown("#### Means, correlations, communalities, and sampling adequacies")
        st.dataframe(model_analysis_styled)
        with st.expander("View interactive standardized loadings heatmap"):
            model_analysis = model_analysis.set_index("VARIABLE")
            model_analysis = model_analysis[[col for col in model_analysis.columns if col.startswith("FACTOR_")]]
            fig_loadings = px.imshow(
                model_analysis,
                text_auto="0.3f",
                aspect="auto",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                labels=dict(x="Factors", y="Variables", color="Standardized loading"),
                title="Factor loading matrix (standardized)"
            )
            fig_loadings.update_xaxes(side="bottom", tickmode="linear", dtick=1)
            fig_loadings.update_yaxes(tickmode="linear", dtick=1)
            fig_loadings.update_layout(
                height=25 * st.session_state.DATA.shape[1]
            )
            st.plotly_chart(fig_loadings, width="stretch")

        model_summary = st.session_state.FACTOR_MODELS.summarize_model(model_name)
        factor_scores = model_summary["scores"]
        df_factor_scores = pd.DataFrame(factor_scores, columns=[f"factor_{idx + 1}"
                                                                for idx in range(factor_scores.shape[1])])
        df_factor_scores_long = df_factor_scores.melt(var_name="Factor", value_name="Score")
        fig_scores_hist = px.histogram(
            df_factor_scores_long,
            x="Score",
            color="Factor",
            title="Factor score distribution",
            marginal="box",
            barmode="overlay",
            opacity=0.8
        )
        st.plotly_chart(fig_scores_hist, width="stretch")
        df_data_with_factor_scores = pd.concat([st.session_state.DATA, df_factor_scores], axis=1)
        col_1, col_2, col_3 = st.columns(3)
        with col_2:
            st.download_button("Download factor scores as CSV file", df_data_with_factor_scores.to_csv(),
                               file_name="factor_scores.csv", width="stretch")

        if st.session_state.FACTOR_MODELS.models[model_name].is_orthogonal_:
            factor_corr_mat = np.eye(st.session_state.FIT_DETAILS[model_name]["number_of_factors"])
        else:
            factor_corr_mat = st.session_state.FACTOR_MODELS.models[model_name].phi_
        df_factor_corr_mat = pd.DataFrame(factor_corr_mat,
                                          columns=[f"factor_{idx + 1}" for idx in range(factor_scores.shape[1])],
                                          index=[f"factor_{idx + 1}" for idx in range(factor_scores.shape[1])])
        fig_factor_corr = px.imshow(
            df_factor_corr_mat,
            text_auto="0.3f",
            aspect="auto",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1,
            title="Factor correlations"
        )
        fig_factor_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
        fig_factor_corr.update_yaxes(tickmode="linear", dtick=1)
        st.plotly_chart(fig_factor_corr, width="stretch")
        col_1, col_2, col_3 = st.columns(3)
        with col_2:
            st.download_button("Download factor correlations as CSV file", df_factor_corr_mat.to_csv(),
                               file_name="factor_correlations.csv", width="stretch")


# Header
col_1, col_2, col_3 = st.columns([1, 3, 1])
with col_2:
    st.image("./images/factor_flow_logo.png", width="stretch")
    with st.container(horizontal_alignment="center"):
        with st.spinner("Loading NLP models...", show_time=True):
            while st.session_state.USE_MODEL_LOADED != 1:
                time.sleep(0.1)
st.subheader("FactorFlow: An LLM-enhanced Visual Workbench for Exploratory Factor Analysis")
st.markdown("Developed by [Justin Philip Tuazon](https://www.linkedin.com/in/justin-philip-tuazon/)")


# Sidebar
st.sidebar.title("Menu")

with st.sidebar.expander("NLP Models", expanded=True):
    st.subheader("Universal Sentence Encoder")

    st.write("Embedder status:")
    if st.session_state.USE_MODEL_LOADED == 1:
        st.badge("Loaded", color="green")
    else:
        st.badge("Loading", color="yellow")

    st.write("Embedder status:")
    if isinstance(st.session_state.EMBEDDINGS, np.ndarray):
        st.badge("Calculated", color="green")
    else:
        if st.session_state.STATEMENTS is None:
            st.badge("No statements loaded", color="yellow")
        else:
            st.badge("Calculating", color="blue")

    st.subheader("Large Language Model")
    st.session_state.CURRENT_LLM_MODEL_ID = st.selectbox(
        "Model in use",
        options=LLM_MODEL_IDS,
        help="Different large language models may give different results. They also have different usage rate limits."
    )
    try:
        llms = st.session_state.LLM_CLIENT.models.list()
        st.badge("Connected", color="green")
    except Exception as e:
        st.badge("Not connected", color="red")

with st.sidebar.expander("Dataset", expanded=True):
    if st.session_state.DATA is None:
        st.warning("You have not uploaded a dataset yet.")
        col_1, col_2, col_3 = st.columns([1, 5, 1])
        with col_2:
            st.button("Upload", width="stretch", disabled=(st.session_state.USE_MODEL_LOADED != 1),
                      on_click=upload_data_dialog)
    else:
        col_1, col_2 = st.columns(2)
        with col_1:
            st.button("Change", width="stretch", on_click=upload_data_dialog)
        with col_2:
            if st.button("Clear", width="stretch"):
                st.session_state.DATA = None
                st.session_state.DATA_NAME = None
                st.session_state.STATEMENTS = None
                st.session_state.STATEMENTS_DF = None
                st.session_state.EMBEDDINGS = None
                st.session_state.SEMANTIC_SIMILARITY_MATRIX = None
                st.session_state.FACTOR_MODELS = None
                st.session_state.FIT_DETAILS = {}
                st.session_state.INTERPRETATIONS = {}
                st.session_state.STATEMENT_TAGS = {}
                clear_embeddings()
                st.rerun()

    if st.session_state.DATA is not None:
        st.write("Dataset name:")
        st.badge(str(st.session_state.DATA_NAME), color="green")
        st.write("Number of observations:")
        st.badge(str(st.session_state.DATA.shape[0]), color="green")
        st.write("Number of manifest variables:")
        st.badge(str(st.session_state.DATA.shape[1]), color="green")
        st.write("With statements:")
        if st.session_state.STATEMENTS is None:
            st.badge("No", color="red")
        else:
            st.badge("Yes", color="green")

        col_1, col_2 = st.columns(2)
        with col_1:
            st.button("Tags", width="stretch", on_click=view_tags_dialog,
                      disabled=(st.session_state.STATEMENTS is None))
        with col_2:
            st.button("Basic stats", width="stretch", on_click=view_data_dialog)

with st.sidebar.expander("Factor Models", expanded=True):
    col_1, col_2 = st.columns(2)
    with col_1:
        st.button("Add", width="stretch", on_click=fit_model_dialog,
                  disabled=(st.session_state.USE_MODEL_LOADED != 1) or (st.session_state.DATA is None))
    with col_2:
        st.button("View", width="stretch", on_click=view_models_dialog,
                  disabled=(st.session_state.DATA is None or len(st.session_state.FACTOR_MODELS.models) == 0))

    st.subheader("Models")
    if st.session_state.DATA is None or len(st.session_state.FACTOR_MODELS.models) == 0:
        st.warning("You have not estimated any factor model yet.")
    elif len(st.session_state.FACTOR_MODELS.models) > 0:
        model_names = sorted(list(st.session_state.FACTOR_MODELS.models.keys()))
        for model_name in model_names:
            st.badge(model_name, color="green")

# Body
tab_overview, tab_dashboard = st.tabs(["Overview", "Dashboard"])

with tab_overview:
    with st.expander("Description", True, icon="📄"):
        st.markdown("""
        FactorFlow is an interactive tool intended to help practitioners perform exploratory factor
        analysis better. Using this tool, users can upload their dataset, fit various factor models, and 
        perform factor rotations. It comes with the following key features or components:
        * Readily available classical rotations (e.g., varimax and more) and traditional visualizations (e.g., 
        correlation heatmap) for core exploratory factor analysis
        * Implementation of pairwise target rotation and interpretability plots from [Pairwise Target Rotation for 
        Factor Models](https://arxiv.org/abs/2409.11525) for going beyond the classical methods
        * Large language model integration for factor model interpretation
        
        Using this tool, practitioners can easily perform exploratory factor analysis and even leverage semantic or 
        arbitrary information for analyzing factor models.
        """)
        col_1, col_2, col_3 = st.columns([1, 5, 1])
        with col_2:
            fig_sample_v_plot = px.scatter(
                st.session_state.SAMPLE_V_DATA,
                x="Prior",
                y="Loading",
                animation_frame="Quality",
                animation_group="Group",
                trendline="lowess"
            )
            fig_sample_v_plot.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000
            fig_sample_v_plot.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 1000
            fig_sample_v_plot.update_layout(
                xaxis_title="Prior Similarity",
                yaxis_title="Semantic Similarity",
                margin=dict(l=50, r=50, t=50, b=50),
                title="Factor Model Interpretability Plot"
            )
            st.plotly_chart(fig_sample_v_plot, width="stretch")
        st.write("Feel free to interact with the sample visualization above to see how the plot changes depending on "
                 "how interpretable the factor model is!")

    with st.expander("Getting started", True, icon="🚀"):
        st.markdown("""
        In general, the user can perform the following steps in order to use this tool:
        1. Upload your dataset in the *Dataset* section of the *Menu*. You can upload two kinds of files: 
            * **The main dataset (CSV file)**. This is the tabular dataset on which the factor models will be fit. Each 
            column must represent a feature and each observation must represent an observation. All data values must be 
            numeric and there must be no missing values. This is **required** to fit a model. The CSV file or raw 
            dataset **should not** have column headers. The tool will automatically label the columns as X1, X2, and 
            so on.
            * **The statements associated with the features (TXT file)**. This is the list of questions or statements 
            associated with each feature in the main dataset. It must be a text file, where statements are separated by 
            linebreaks - consecutive lines with one statement per line. The order of the statements must match the order 
            of the columns in the main dataset (i.e., the first statement must correspond to the first feature). This 
            is an **optional** input, and will be used only if you select "semantics" for the prior in pairwise target 
            rotation.
            * **The tags associated with each statement**. You can add tags to each statement to help visualize 
            interpretability. To do so, click *Tags* under *Dataset*. This is optional.
        2. Fit one or more factor models in the *Models* section of the *Menu*. Each model will use the same main 
        dataset. You can add or remove as many factor models as you need to. You can click the model name in order to 
        see more details about how the model was fit (e.g., number of factors, rotation method, fitting algorithm).
            * When uploading a CSV file for a custom prior matrix, make sure that the matrix is symmetric and that 
            the number of rows (or columns) matches the number of manifest variables (i.e., columns in the main 
            dataset). Also, all entries must be either a number or left blank.
            * Note that the tool standardizes (i.e., subtract mean and divide by standard deviation) each manifest 
            variable prior to fitting. This means that the loadings provided are standardized loadings (i.e., 
            **correlations** with the factors).
        3. Proceed to the *Dashboard* tab and examine the loadings and visualizations available for each model. You can 
        choose to display only one model to focus on a single factor model but you can also display 2 factor 
        models at the same time for comparisons.
            * If you want to view one factor model at a time in detail instead, you can go to *View* under 
            *Factor Models*.
        
        Note that you can **download** most datasets, tables, and visualizations shown in this tool. **A video 
        walkthrough of the tool is in the works**.
        """)

        st.markdown("""
        Sample datasets are available 
        [here](https://drive.google.com/drive/folders/1nc-pZFM5JdxmMrqE_QJyf03DLTEoEH0X?usp=sharing). 
        Although classical rotations and traditional visualizations are made available, this tool was made partially 
        to make pairwise target rotation accessible. As such, you may want to read the 
        [paper](https://arxiv.org/abs/2409.11525) to understand more about how you can use this tool.
        """)

    with st.expander("Notes", True, icon="📌"):
        st.markdown("""
        * Right now, the tool does not support a correlation matrix as the main dataset and does not support
        polychoric correlations. These will be added in the future.
        * The Universal Sentence Encoder is the only embedding model supported right now.
        * This tool is made available under the GNU General Public License v3.0.
        * The code repository for this tool can be found 
        [here](https://github.com/jptuazon/st-simple-churn-dashboard/blob/main/app.py).
        """)

    with st.expander("Dependencies", False, icon="🔗"):
        st.write("This tool uses several third-party Python packages or dependences, which are listed below.")
        with open("./requirements.txt", "r") as f:
            for line in f:
                st.markdown(f"* {line}")

with tab_dashboard:
    st.markdown(":bulb: If your screen is not wide enough for the horizontal layout, "
                "consider temporarily hiding the *Menu* sidebar. You can also hide or show columns in tables.")
    if st.session_state.DATA is None or len(st.session_state.FACTOR_MODELS.models.keys()) == 0:
        st.warning("Fit a factor model first.")
    else:
        col_1, col_2 = st.columns(2)
        with col_1:
            selected_models = st.multiselect(
                "Select models to examine", options=sorted(list(st.session_state.FACTOR_MODELS.models.keys())),
                help="You can choose up to 2 models at a time.",
                max_selections=2
            )

        st.space()
        if len(selected_models) == 0:
            st.warning("Choose at least one model.")
        else:
            cols = st.columns(len(selected_models), border=True)
            multisets = [
                st.session_state.FACTOR_MODELS.generate_multiset(model_name)
                for model_name in selected_models
            ]
            model_analyses = [
                st.session_state.FACTOR_MODELS.analyze_model(model_name)
                for model_name in selected_models
            ]

            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]
                    multiset = multisets[i]
                    model_analysis = model_analyses[i]
                    loadings_only = model_analysis[["variable"] + [col for col in model_analysis.columns
                                                   if col.startswith("factor_")]]

                    st.subheader(model_name)
                    st.divider()

                    st.badge("Fit details", color="blue")
                    fit_details = st.session_state.FIT_DETAILS[model_name]

                    col_1, col_2, col_3, col_4 = st.columns([3, 3, 3, 3])
                    with col_1:
                        st.write("Factors")
                        st.badge(str(fit_details["number_of_factors"]), color="green")
                    with col_2:
                        st.write("Rotation")
                        st.badge(str(fit_details["rotation"]).capitalize(), color="green")
                    with col_3:
                        st.write("Prior type")
                        st.badge(str(fit_details["prior"]), color="green")
                    with col_4:
                        st.write("V index")
                        v = st.session_state.FACTOR_MODELS.calculate_v_index(model_name)
                        v = np.round(v, 5) if v is not None else None
                        st.badge(str(v), color="green")

                    st.space()

                    st.badge("Communalities and adequacies", color="blue")
                    comm_and_adeq = model_analysis[["variable", "communality", "kmo_msa"]]
                    if st.session_state.STATEMENTS_DF is not None:
                        comm_and_adeq = pd.merge(left=comm_and_adeq, right=st.session_state.STATEMENTS_DF,
                                                 left_on="variable", right_on="Variable", how="left")
                        comm_and_adeq = comm_and_adeq[["variable", "Statement", "communality", "kmo_msa"]]
                    comm_and_adeq.columns = [col.upper() for col in comm_and_adeq.columns]
                    comm_and_adeq_styled = comm_and_adeq.reset_index(drop=True)
                    comm_and_adeq_styled = comm_and_adeq_styled.style.background_gradient(
                        cmap="Purples", axis=0, subset=["COMMUNALITY", "KMO_MSA"], vmin=0, vmax=1.0
                    )
                    st.dataframe(comm_and_adeq_styled, hide_index=True, key=f"{model_name}_comm_and_adeq")

                    st.badge("Interpretability plot", color="blue")
                    similarity_type = ("Semantic Similarity"
                                       if st.session_state.FIT_DETAILS[model_name]["prior"] == "Semantics"
                                       else "Prior Similarity")

                    if multiset is None:
                        df_multiset = pd.DataFrame({
                            similarity_type: [],
                            "Loading Similarity": []
                        })
                    else:
                        df_multiset = pd.DataFrame({
                            similarity_type: [item[0] for item in multiset],
                            "Loading Similarity": [item[1] for item in multiset]
                        })

                    fig_v_plot = px.scatter(
                        df_multiset,
                        x=similarity_type,
                        y="Loading Similarity",
                        trendline="lowess",
                        title=f"{similarity_type} vs Loading Similarity",
                        subtitle=f"V = "
                                 f"{st.session_state.FACTOR_MODELS.calculate_v_index(model_name)}"
                    )

                    st.plotly_chart(fig_v_plot, width="stretch", key=f"{model_name}_v_plot")

                    st.badge("Factor loadings", color="blue")
                    thresh = st.slider(
                        "Absolute threshold", min_value=0.0, max_value=1.0, value=0.35,
                        help="""
                        This discretizes the loadings such that a manifest variable is "included" in the 
                        factor (and its interpretation) if and only if the absolute value of the loading 
                        is at least the threshold.
                        """, key=f"{model_name}_thresh_slider"
                    )
                    col_1, col_2 = st.columns(2)
                    with col_1:
                        show_raw = st.checkbox("Show original loadings instead?",
                                               key=f"{model_name}_show_raw_instead")
                    with col_2:
                        sort_by_variable = st.checkbox(
                            "Sort by variable name instead?", key=f"{model_name}_sort_by_var",
                            help="By default, the manifest variables are sorted in descending order in terms of "
                                 "their largest loadings. Clicking this will sort them by their name instead."
                        )

                    df_loadings = loadings_only.copy()
                    if sort_by_variable:
                        df_loadings.sort_index(inplace=True)
                    df_loadings = df_loadings.set_index("variable")
                    factor_cols = [col for col in df_loadings.columns if col.startswith("factor_")]

                    df_loadings_discretized = df_loadings.copy()
                    df_loadings_discretized[factor_cols] = df_loadings_discretized[factor_cols].abs().ge(
                        float(thresh)
                    ).astype(int)
                    if not show_raw:
                        df_loadings = df_loadings_discretized.copy()

                    fig_loadings = px.imshow(
                        df_loadings,
                        text_auto="0.3f" if show_raw else "0",
                        aspect="auto",
                        color_continuous_scale="RdBu" if show_raw else "Purples",
                        color_continuous_midpoint=0 if show_raw else 0.5,
                        labels=dict(x="Factors", y="Variables",
                                    color="Loading" if show_raw else "Included"),
                        title="Factor loading matrix"
                    )
                    fig_loadings.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                    fig_loadings.update_yaxes(tickmode="linear", dtick=1)
                    fig_loadings.update_layout(
                        height=25 * st.session_state.DATA.shape[1]
                    )
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
                                       file_name="factor_loadings.csv", width="stretch")
                    st.space()
                    with st.expander("View variables / statements associated with each factor"):
                        for idx, factor in enumerate(df_loadings_discretized.columns):
                            factor_loadings = df_loadings_discretized[factor]
                            variables = df_loadings_discretized[factor_loadings == 1].index.tolist()
                            st.markdown(f"#### {factor}")
                            st.write(f"{len(variables)} manifest variable{'s' if len(variables) > 1 else ''}")
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
                    st.space()

                    st.badge("Factor breakdown", color="blue")

                    df_tags_breakdown = compute_tags_breakdown(loadings_only)
                    fig_tags_breakdown = px.bar(
                        df_tags_breakdown,
                        x="Factor",
                        y="Sum of Squared Loadings",
                        color="Tag",
                        barmode="stack",
                        title="Sum of squared loadings per factor"
                    )
                    st.plotly_chart(fig_tags_breakdown, width="stretch", key=f"{model_name}_tags_breakdown")

                    st.badge("Interpretation", color="blue")
                    if st.session_state.STATEMENTS is None:
                        st.warning("The associated statements for the variables were not provided.")
                    else:
                        st.write("Interpretations are generated using the groupings defined by the "
                                 "absolute threshold above.")
                        col_1, col_2 = st.columns(2)
                        with col_2:
                            if st.button("Generate new interpretation",
                                         key=f"{model_name}_interpret_factor_model", width="stretch"):
                                st.session_state.INTERPRETATIONS[model_name] = interpret_factor_model(
                                    df_loadings_discretized
                                )
                        if st.session_state.INTERPRETATIONS[model_name][0] is not None:
                            st.space()
                            if st.session_state.INTERPRETATIONS[model_name][0] != "Error":
                                st.write(f"Generated by {st.session_state.INTERPRETATIONS[model_name][0]}")
                            st.write(st.session_state.INTERPRETATIONS[model_name][1])
