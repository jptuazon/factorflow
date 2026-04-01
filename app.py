# Copyright 2026 Justin Philip Tuazon

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

# FactorFlow V2.0.2
# https://factorflow-efa.streamlit.app/

import warnings
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
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_extras.card_selector import card_selector
from streamlit_extras.floating_button import floating_button
from streamlit_extras.scroll_to_element import scroll_to_element
from pygwalker.api.streamlit import StreamlitRenderer


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
if "SCROLL_COUNTER" not in st.session_state:
    st.session_state.SCROLL_COUNTER = 0

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
    st.session_state.FACTOR_MODELS = {}

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

if "manifest_vars" not in st.session_state:
    st.session_state.manifest_vars = None

if "SAMPLE_V_DATA" not in st.session_state:
    st.session_state.SAMPLE_V_DATA = pd.read_csv("./sample_data/sample_v_data.csv")

if "CURRENT_LLM_MODEL_ID" not in st.session_state:
    st.session_state.CURRENT_LLM_MODEL_ID = None

if "INTERPRETATIONS" not in st.session_state:
    st.session_state.INTERPRETATIONS = {}

if "VARIABLE_TAGS" not in st.session_state:
    st.session_state.VARIABLE_TAGS = {}


# Fit factor model
@st.dialog("Fit a new model", dismissible=False)
def fit_factor_model():
    if st.session_state.FIT_MODEL == "Yes":
        with st.spinner("Fitting the factor model...", width="stretch", show_time=True):
            model_name = str(st.session_state.model_name)
            number_of_factors = int(st.session_state.number_of_factors)
            rotation = None if st.session_state.rotation == "None" else str(st.session_state.rotation).lower()
            prior_matrix = st.session_state.prior_matrix
            manifest_vars = st.session_state.manifest_vars

            if rotation == "equamax":
                rot_kwargs = {
                    "kappa": number_of_factors / (2 * len(manifest_vars))
                }
            else:
                rot_kwargs = None

            st.session_state.FACTOR_MODELS[model_name] = InterpretableFA(
                st.session_state.DATA[manifest_vars]
            )
            st.session_state.FACTOR_MODELS[model_name].fit_factor_model(
                model_name,
                number_of_factors,
                rotation,
                prior_matrix,
                rotation_kwargs=rot_kwargs
            )

            st.session_state.FIT_DETAILS[model_name] = {
                "number_of_factors": number_of_factors,
                "rotation": rotation,
                "prior": st.session_state.prior,
                "manifest_vars": manifest_vars
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


if st.session_state.FIT_MODEL == "Yes":
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
                    - Format as:
                      Statement 1: [full statement]
                      Statement 2: [full statement]
                    - Preserve the original wording exactly (do NOT paraphrase).
                    - Number statements in the order given in the input.
                    
                    Description Requirements:
                    - Avoid surface-level or generic interpretations.
                    - Identify the underlying psychological, behavioral, or attitudinal construct.
                    - Prefer abstract constructs over literal summaries of statements.
                    
                    Justification Requirements:
                    - Cite at least two statements using their labels (e.g., "Statement 1").
                    - Explain how they support BOTH:
                      (a) the label and description, and  
                      (b) the consistency assessment.
                    - Go beyond restating. Provide reasoning.
                    
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
                    - "No statements" and "Not applicable" are used correctly when required.
                    - No full statements appear outside the Statements section.
                    - All references use "Statement X" format.
                    - Formatting exactly matches the template.
                    - All rules and requirements are followed.
                    
                    Output Template (APPLY TO EVERY FACTOR WITHOUT EXCEPTION):
                    
                    **factor_X**
                    
                    • **Statements**:
                      Statement 1: [full statement]  
                      Statement 2: [full statement]
                    
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
    except Exception:
        st.toast("❌ Cannot use current LLM. Please try again in a while or use a different model.")
        return "Error", "Rate limit reached. Please try again in a while or use a different model."


def interpret_factor_model(df_discretized_loadings):
    input_for_llm = ""

    for idx, factor in enumerate(df_discretized_loadings.columns):
        factor_loadings = df_discretized_loadings[factor]
        variables = df_discretized_loadings[factor_loadings == 1].index.tolist()

        input_for_llm += f"\n{factor}:\n"
        if len(variables) == 0:
            input_for_llm += "- No statements."
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
    initial_sidebar_state="auto",
    menu_items={
        "About": """
        * Version Number: 2.0.2
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
    # CSS for the square
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
    # Render HTML + Label
    st.markdown(f"{square_html}<span style='vertical-align: middle;'>{label}</span>",
                unsafe_allow_html=True)


def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"


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


def process_prior_matrix(prior_matrix, rotation, manifest_vars):
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


@st.dialog(":material/upload: Upload dataset", width="medium")
def upload_data_dialog():
    st.badge(":material/info: Ensure that you have read *Getting started* in the *Overview* tab before "
             "proceeding.",
             color="blue")
    df_data = None
    data_file_name = None
    statements = None
    statements_file_name = None

    can_proceed = True

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
            use_statements = st.checkbox("Upload statements or questions associated with the manifest variables?")
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
        col_1, col_2 = st.columns([15, 3])
        with col_2:
            if st.button("Confirm", width="stretch"):
                st.session_state.DATA = df_data
                st.session_state.DATA_NAME = data_file_name
                st.session_state.STATEMENTS = statements
                st.session_state.STATEMENTS_NAME = statements_file_name
                st.session_state.STATEMENTS_DF = pd.DataFrame({
                    "Variable": [f"X{idx + 1}" for idx in range(df_data.shape[1])],
                    "Statement": statements
                })
                st.session_state.VARIABLE_TAGS = {}
                for idx in range(df_data.shape[1]):
                    st.session_state.VARIABLE_TAGS[f"X{idx + 1}"] = []
                st.rerun()


@st.dialog(":material/shoppingmode: Manifest variable tags", width="large")
def view_tags_dialog():
    st.write("""
    Tags are a priori labellings of variables or statements. For instance, you can tag the statements "I am loyal to 
    brands I have used before." and "I associate products with memories." with "Nostalgia". Each variable can have 
    zero or more tags. These tags are then used to summarize the "breakdown" of a factor in the *Dashboard*.
    """)

    df_tags = pd.DataFrame([
        {"Variable": variable, "Tags": ", ".join(tags)}
        for variable, tags in st.session_state.VARIABLE_TAGS.items()
    ])
    df_tags.index = [f"X{int(idx + 1)}" for idx in df_tags.index]
    df_tags = pd.merge(left=df_tags, right=st.session_state.STATEMENTS_DF, on="Variable", how="left")
    df_tags = df_tags[["Variable", "Statement", "Tags"]]
    df_tags.sort_values(by=["Statement", "Variable"], ascending=[True, True], inplace=True)
    st.dataframe(df_tags)

    can_proceed = True

    st.space()
    col_1, col_2 = st.columns([6, 1])
    with col_2:
        if st.button("Clear all tags", width="stretch"):
            st.session_state.VARIABLE_TAGS = {}
            st.rerun()

    st.markdown("## Edit tags")

    input_type = st.selectbox(
        "Input type",
        options=["Manual", "Upload"]
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
            options=choices
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
            placeholder="Selecting no tag will remove all tags.."
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
        col_1, col_2 = st.columns([6, 1])
        with col_2:
            if st.button("Confirm", width="stretch"):
                if input_type == "Manual":
                    st.session_state.VARIABLE_TAGS[current_variable] = new_tags
                else:
                    st.dataframe(df_new_tags)
                    for idx, row in df_new_tags.iterrows():
                        variable = row["variable"]
                        tags = row["tags"].split(",") if not pd.isna(row["tags"]) else []
                        tags = [tag.strip() for tag in tags if tag.strip() != ""] if len(tags) > 0 else tags
                        st.session_state.VARIABLE_TAGS[variable] = tags
                st.rerun()


@st.dialog(":material/analytics: View basic stats", width="large")
def view_data_dialog():
    with st.expander("Raw data"):
        st.dataframe(st.session_state.DATA)
        st.space()

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
        st.space()

    with st.expander("Summary statistics"):
        st.dataframe(st.session_state.DATA.describe())
        st.space()

    with st.expander("Correlation matrix"):
        corr_mat = st.session_state.DATA.corr()
        fig_corr = px.imshow(
            corr_mat,
            text_auto="0.3f",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Sample Correlation Matrix"
        )
        fig_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
        fig_corr.update_yaxes(tickmode="linear", dtick=1)
        fig_corr.update_layout(
            height=min(900, max(50 * st.session_state.DATA.shape[1], 300))
        )
        st.plotly_chart(fig_corr, width="stretch")
        st.dataframe(corr_mat)
        st.space()

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
                    color_continuous_scale="Blues",
                    zmin=0, zmax=1,
                    title="Semantic Similarity Matrix"
                )
                fig_semantic.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                fig_semantic.update_yaxes(tickmode="linear", dtick=1)
                fig_semantic.update_layout(
                    height=min(900, max(50 * st.session_state.DATA.shape[1], 300))
                )
                st.plotly_chart(fig_semantic, width="stretch")
                st.dataframe(semantic_similarity_mat)
            else:
                st.write(":hourglass_flowing_sand: Loading...")
        st.space()


@st.dialog(":material/add_chart: Fit a new factor model", width="large", dismissible=False)
def fit_model_dialog():
    model_name = None
    number_of_factors = None
    rotation = None
    prior = None
    prior_matrix = None
    manifest_vars = None

    st.badge("""
    :material/info: For the model name, it is recommended to follow this format: 
    [rotation]\\_[number of factors]\\_[prior matrix description] (e.g., "priorimax_3_semantics", "varimax_5_custom").
    """, color="blue")

    can_proceed = True

    col_1, col_2 = st.columns(2)
    with col_1:
        model_name = st.text_input(label="Model name", value=f"model_{len(st.session_state.FACTOR_MODELS) + 1}",
                                   help="This must be unique.", width="stretch")
    with col_2:
        number_of_factors = st.number_input(label="Number of factors", placeholder="Enter a positive number...",
                                            value=3, min_value=1, max_value=(st.session_state.DATA.shape[1] - 1),
                                            help="The number of factors cannot exceed the number of manifest "
                                                 "variables.", width="stretch")
    manifest_vars = st.multiselect(
        "Manifest variables",
        options=[f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])],
        help="Only these manifest variables will be considered in the factor model."
    )
    col_1, col_2 = st.columns(2)
    with col_1:
        rotation = st.selectbox(label="Rotation", placeholder="Select a rotation method...",
                                options=ROTATIONS, help="Priorimax, Varimax, Oblimax, Quartimax, and Equamax are "
                                                        "orthogonal rotations. Promax, Oblimin, and Quartimin are "
                                                        "oblique rotations.", width="stretch")
    with col_2:
        prior = st.selectbox(label="Prior matrix", placeholder="Specify the prior matrix for priorimax...",
                             options=("Semantics", "Grouped", "Custom", "None"),
                             help="""
                             "Semantics" uses the semantic similarity matrix. "Custom" lets you specify the exact 
                             matrix by uploading a CSV file. This is required when using the priorimax rotation and 
                             optional for other rotation methods.""", width="stretch")
        show_prior_matrix = st.checkbox("Show exact prior matrix?")

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
                        color_continuous_scale="Blues",
                        zmin=0, zmax=1
                    )
                    fig_semantic.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                    fig_semantic.update_yaxes(tickmode="linear", dtick=1)
                    fig_semantic.update_layout(
                        height=min(900, max(50 * len(manifest_vars), 300)),
                        title="The semantic similarity matrix will be used as the prior matrix."
                    )
                    st.plotly_chart(fig_semantic, width="stretch")
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
                            st.multiselect(label=f"Grouping {factor_number + 1}", options=group_choices)
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
                        color_continuous_scale="Blues",
                        zmin=0, zmax=1
                    )
                    fig_prior.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                    fig_prior.update_yaxes(tickmode="linear", dtick=1)
                    fig_prior.update_layout(
                        height=min(900, max(50 * prior_matrix.shape[1], 300)),
                        title="This grouping matrix will be used as the prior matrix."
                    )
                    st.plotly_chart(fig_prior, width="stretch")
        elif prior == "Custom":
            prior_matrix = st.file_uploader(label="Upload CSV file for the prior matrix", type="csv", width="stretch",
                                            disabled=(prior != "Custom"))
            if prior_matrix is not None:
                prior_matrix = pd.read_csv(prior_matrix, header=None).to_numpy()

                check = process_prior_matrix(prior_matrix, rotation, manifest_vars)
                if check["pass"]:
                    prior_matrix = pd.DataFrame(prior_matrix, index=manifest_vars, columns=manifest_vars)
                    if show_prior_matrix:
                        fig_prior = px.imshow(
                            prior_matrix,
                            text_auto="0.3f",
                            aspect="auto",
                            color_continuous_scale="Blues",
                            zmin=0, zmax=1
                        )
                        fig_prior.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                        fig_prior.update_yaxes(tickmode="linear", dtick=1)
                        fig_prior.update_layout(
                            height=min(900, max(50 * len(manifest_vars), 300)),
                            title="This custom matrix will be used as the prior matrix."
                        )
                        st.plotly_chart(fig_prior, width="stretch")
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
                st.dataframe(manifest_statements)
                can_proceed = False
        elif prior == "None":
            prior_matrix = None

    st.space()
    col_1, col_2, col_3 = st.columns([12, 3, 3])
    with col_3:
        if st.button("Cancel", width="stretch"):
            st.rerun()
    if can_proceed:
        with col_2:
            if st.button("Fit model", width="stretch", type="primary"):
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
                        prior_matrix = prior_matrix.to_numpy()
                else:
                    prior_matrix = None

                if not any_failed:
                    st.session_state.model_name = model_name
                    st.session_state.number_of_factors = number_of_factors
                    st.session_state.rotation = rotation
                    st.session_state.prior_matrix = prior_matrix
                    st.session_state.prior = prior
                    st.session_state.manifest_vars = manifest_vars
                    st.session_state.FIT_MODEL = "Yes"
                    st.rerun()


def delete_factor_model(model_name):
    del st.session_state.FACTOR_MODELS[model_name]
    del st.session_state.FIT_DETAILS[model_name]
    if model_name in st.session_state.INTERPRETATIONS:
        del st.session_state.INTERPRETATIONS[model_name]


@st.dialog(":material/bar_chart: View factor models", width="large", on_dismiss="rerun")
def view_models_dialog():
    model_name = st.selectbox("Choose a model", options=sorted(list(st.session_state.FACTOR_MODELS.keys())),
                              width="stretch")

    if model_name is None:
        st.error("There are no factor models available.")
    else:
        col_1, col_2 = st.columns([21, 3])
        with col_2:
            st.button("Delete model", on_click=delete_factor_model, args=(model_name,),
                      width="stretch")

        st.subheader("Fit details")

        factor_model = st.session_state.FACTOR_MODELS[model_name].models[model_name]
        fit_details = st.session_state.FIT_DETAILS[model_name]

        col_1, col_2, col_3, col_4, col_5 = st.columns(5)
        with col_1:
            st.caption("Number of manifest variables")
            st.badge(str(len(fit_details["manifest_vars"])), color="green")
        with col_2:
            st.caption("Number of factors")
            st.badge(str(fit_details["number_of_factors"]), color="green")
        with col_3:
            st.caption("Rotation")
            st.badge(str(fit_details["rotation"]).capitalize(), color="green")
        with col_4:
            st.caption("Prior type")
            st.badge(str(fit_details["prior"]), color="green")
        with col_5:
            st.caption("V-index")
            v = st.session_state.FACTOR_MODELS[model_name].calculate_v_index(model_name)
            v = np.round(v, 5) if v is not None else None
            st.badge(str(v), color="green")

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
                    color_continuous_scale="Blues",
                    zmin=0, zmax=1
                )
                fig_prior.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                fig_prior.update_yaxes(tickmode="linear", dtick=1)
                fig_prior.update_layout(
                    height=min(900, max(50 * len(fit_details["manifest_vars"]), 300)),
                    title="Prior Matrix"
                )
                st.plotly_chart(fig_prior, width="stretch")
                col_1, col_2, col_3 = st.columns(3)
                with col_2:
                    st.download_button("Download prior matrix as CSV file", df_prior_matrix.to_csv(),
                                       file_name="prior_matrix.csv", width="stretch")
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
            cmap="RdBu", axis=None, subset=[col for col in model_analysis.columns if col.startswith("factor_")],
            vmin=-1.0, vmax=1.0
        ).background_gradient(
            cmap="Purples", axis=0, subset=["COMMUNALITY", "KMO_MSA"], vmin=0, vmax=1.0
        )

        st.caption("Means, correlations, communalities, and sampling adequacies")
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
                title="Factor Loading Matrix (Standardized)"
            )
            fig_loadings.update_xaxes(side="bottom", tickmode="linear", dtick=1)
            fig_loadings.update_yaxes(tickmode="linear", dtick=1)
            fig_loadings.update_layout(
                height=min(900, max(50 * len(fit_details["manifest_vars"]), 300))
            )
            st.plotly_chart(fig_loadings, width="stretch")

        st.space()

        model_summary = st.session_state.FACTOR_MODELS[model_name].summarize_model(model_name)
        factor_scores = model_summary["scores"]
        df_factor_scores = pd.DataFrame(factor_scores, columns=[f"factor_{idx + 1}"
                                                                for idx in range(factor_scores.shape[1])])
        df_factor_scores_long = df_factor_scores.melt(var_name="Factor", value_name="Score")
        fig_scores_hist = px.histogram(
            df_factor_scores_long,
            x="Score",
            color="Factor",
            marginal="box",
            barmode="overlay",
            opacity=0.8
        )
        st.caption("Factor score distribution")
        st.plotly_chart(fig_scores_hist, width="stretch")
        df_data_with_factor_scores = pd.concat([st.session_state.DATA, df_factor_scores], axis=1)
        col_1, col_2, col_3 = st.columns(3)
        with col_2:
            st.download_button("Download factor scores as CSV file", df_data_with_factor_scores.to_csv(),
                               file_name="factor_scores.csv", width="stretch")

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
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
        )
        fig_factor_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
        fig_factor_corr.update_yaxes(tickmode="linear", dtick=1)
        st.caption("Factor correlations")
        st.plotly_chart(fig_factor_corr, width="stretch")
        col_1, col_2, col_3 = st.columns(3)
        with col_2:
            st.download_button("Download factor correlations as CSV file", df_factor_corr_mat.to_csv(),
                               file_name="factor_correlations.csv", width="stretch")

        st.space()


# Header
st.markdown("""
    <style>
        .st-key-load_use_model iframe, .st-key-get_embeddings iframe {
            height: 0px;
            background-color: rgba(0,0,0,0) !important;
        }
    </style>
""", unsafe_allow_html=True)

col_1, col_2, col_3 = st.columns([1, 3, 1])
with col_2:
    st.image("./images/factor_flow_logo.png", width="stretch")
    with st.container(horizontal_alignment="center"):
        with st.spinner("Loading NLP models...", show_time=True):
            while st.session_state.USE_MODEL_LOADED != 1:
                time.sleep(0.1)

st.session_state.SCROLL_COUNTER = 1 - st.session_state.SCROLL_COUNTER
with st.container(key=f"app_title_{st.session_state.SCROLL_COUNTER}"):
    st.subheader("FactorFlow: An LLM-enhanced Visual Workbench for Exploratory Factor Analysis")
st.markdown("Developed by [Justin Philip Tuazon](https://www.linkedin.com/in/justin-philip-tuazon/)")

# Sidebar
st.sidebar.title(":material/menu: Menu")

with st.sidebar:
    show_floating_top = st.checkbox("""Show "Back to Top" button""", value=True)

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
            label_visibility="collapsed"
        )

        col_label, col_help = st.columns([0.9, 0.1])
        with col_label:
            st.caption("LLM temperature:")
        with col_help:
            st.markdown("", help="Larger values encourage randomness and creativity, while "
                                 "smaller values encourage determinism and focus. For more consistent interpretations "
                                 "and formatting, choose a value not greater than 0.1.")
        llm_temp = st.slider("LLM temperature:", min_value=0.0, max_value=0.5, step=0.05, value=0.1,
                             label_visibility="collapsed")
    else:
        st.error("Failed to connect. Please refresh.")

with st.sidebar.expander("Data", icon=":material/dataset:", expanded=True):
    if st.session_state.DATA is None:
        st.warning("You have not uploaded a dataset yet.")
        col_1, col_2, col_3 = st.columns([1, 5, 1])
        with col_2:
            st.button("Upload", width="stretch", disabled=(st.session_state.USE_MODEL_LOADED != 1),
                      on_click=upload_data_dialog)
    else:
        st.subheader("Dataset")
        st.text_input("Dataset", value=str(st.session_state.DATA_NAME),
                      label_visibility="collapsed", disabled=True)

        st.subheader("Statements")
        st.text_input("Statements name", value=str(st.session_state.STATEMENTS_NAME),
                      label_visibility="collapsed", disabled=True)

        col_1, col_2 = st.columns(2)
        with col_1:
            st.caption("Variables")
            st.write(str(st.session_state.DATA.shape[1]))
        with col_2:
            st.caption("Observations")
            st.write(f"{st.session_state.DATA.shape[0]:,}")

        col_1, col_2 = st.columns(2)
        with col_1:
            st.button("Stats", width="stretch", type="primary", on_click=view_data_dialog)
        with col_2:
            st.button("Tags", width="stretch", on_click=view_tags_dialog)

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
                st.session_state.FACTOR_MODELS = {}
                st.session_state.FIT_DETAILS = {}
                st.session_state.INTERPRETATIONS = {}
                st.session_state.VARIABLE_TAGS = {}
                clear_embeddings()
                st.rerun()

with st.sidebar.expander("Factor Models", icon=":material/function:", expanded=True):
    col_1, col_2 = st.columns(2)
    with col_1:
        st.button("Add", width="stretch", on_click=fit_model_dialog, type="primary",
                  disabled=(st.session_state.USE_MODEL_LOADED != 1) or (st.session_state.DATA is None))
    with col_2:
        st.button("View", width="stretch", on_click=view_models_dialog,
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

# Body
tab_overview, tab_diagnostics, tab_dashboard = st.tabs([
    ":material/home: Overview",
    ":material/data_thresholding: Diagnostics",
    ":material/dashboard: Dashboard"
])

with tab_overview:
    with st.expander("Description", True, icon=":material/description:"):
        st.markdown("""
        FactorFlow is an interactive tool intended to help practitioners perform exploratory factor
        analysis better. Using this tool, users can upload their dataset, fit various factor models, and 
        perform factor rotations. It comes with the following key features or components:
        * Readily available classical rotations (e.g., varimax and more) and traditional visualizations (e.g., 
        correlation heatmap) for core exploratory factor analysis
        * Implementation of pairwise target rotation and interpretability plots from [Pairwise Target Rotation for 
        Factor Models](https://arxiv.org/abs/2409.11525) for going beyond the classical methods
        * Large language model integration for factor model interpretation
        * Multiple tabs available for diagnostics, deep dives, and comparisons
        
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
            fig_sample_v_plot.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
            fig_sample_v_plot.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 1000
            fig_sample_v_plot.update_layout(
                xaxis_title="Prior Similarity",
                yaxis_title="Semantic Similarity",
                margin=dict(l=50, r=50, t=50, b=50),
                title="Factor Model Interpretability Plot"
            )
            fig_sample_v_plot.update_layout(
                yaxis=dict(
                    visible=True,
                    showticklabels=True,
                    showline=False,
                    showgrid=False,
                    zeroline=False
                )
            )
            st.plotly_chart(fig_sample_v_plot, width="stretch")
        st.write("Feel free to interact with the sample visualization above to see how the plot changes depending on "
                 "how interpretable the factor model is!")

    with st.expander("Getting started", True, icon=":material/rocket_launch:"):
        st.markdown("""
        In general, you can follow these steps to use FactorFlow.
        """)

        getting_started_step = card_selector(
            [
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
                    title="4. Analyze",
                    description="Examine your models.",
                ),
                dict(
                    icon=":material/export_notes:",
                    title="5. Export",
                    description="Download the results.",
                )
            ],
            default=0,
            key="how_to_guide"
        )

        if getting_started_step == 0:
            st.markdown("""
            Upload your dataset in the *Dataset* section of the *Menu*. You can upload three kinds of files: 
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
            * **The tags associated with each variable**. You can add tags to each variable or statement to help 
            visualize interpretability. To do so, click *Tags* under *Dataset*. This is optional.
            
            **On Statements**. Ideally, statements should not be too long but they should also be "complete" (e.g., 
            a full sentence). However, it is also possible to have just "regular" one or two-word variable names such 
            as "height" and so on. In such cases though, semantic similarities may not be as meaningful.
            
            **Statements vs Tags**. A variable can have at most one statement. It is usually the "question" for the 
            variable. No two variables can have the same statement. On the other hand, a variable can have zero or 
            more tags, and tags do not have to be unique across variables.
            """)
        elif getting_started_step == 1:
            st.markdown("""
            Explore your dataset by going to the *Diagnostics* tab. For instance, you might want to examine the 
            communalities or you might want to determine the optimal number of factors. There is also an interactive 
            visualizer available if you want to manually explore the raw dataset yourself. Otherwise, you can go to 
            *Basic stats* under *Dataset* in the menu to see some readily available summary statistics.
            """)
        elif getting_started_step == 2:
            st.markdown("""
            Fit one or more factor models in the *Models* section of the *Menu*. Each model will use the same main 
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
        elif getting_started_step == 3:
            st.markdown("""
            Proceed to the *Dashboard* tab and examine the loadings and visualizations available for each model. You can 
            choose to display only one model to focus on a single factor model but you can also display 2 factor 
            models at the same time for comparisons. If you want to view one factor model at a time in detail instead, 
            you can go to *View* under *Factor Models*.
            """)
        elif getting_started_step == 4:
            st.markdown("""
            You can **download** most tables, figures, and visualizations in this tool:
            * For tables, hovering on them triggers a download button to show at the top right of the table.
            * For most figures, there are dedicated download buttons that you can click.
            * For most visualizations, hovering on them triggers the control panel to show at the top right of the 
            chart, where you can find the download button. For some others, you can right-click on the chart and 
            click "Save image as..." or "Copy image".
            """)

    with st.expander("Notes", True, icon=":material/pinboard:"):
        st.markdown("""
        ###### Examples 
        * Sample datasets and files are available 
        [here](https://drive.google.com/drive/folders/1nc-pZFM5JdxmMrqE_QJyf03DLTEoEH0X?usp=sharing).
        * A video walkthrough of the tool is in the works.
        * Although classical rotations and traditional visualizations are made available, this tool was made partially
        to make pairwise target rotation accessible. As such, you may want to read the 
        [paper](https://arxiv.org/abs/2409.11525) to understand more about how you can use this tool.

        ###### Limitations and Future Releases
        * The tool currently does not support a correlation matrix as the main dataset and polychoric correlations. 
        These will be added in the future.
        * The Universal Sentence Encoder is the only embedding model supported right now.
        
        ###### Additional Information
        * You can switch between Dark and Light modes by clicking the Settings icon at the top right of the page.
        * The code repository for this tool can be found [here](https://github.com/jptuazon/factorflow).
        * FactorFlow is made available under the GNU General Public License v3.0.
        """)

with tab_diagnostics:
    if st.session_state.DATA is None:
        st.warning("Upload a dataset first.")
    else:
        diagnostics_sub_tab = st.radio(
            "Choose what to examnie",
            ["Factor model goodness-of-fit", "Raw data explorer"],
            key="diag_sub_tab",
            horizontal=True
        )
        st.space("small")

        if diagnostics_sub_tab == "Factor model goodness-of-fit":
            manifest_vars = st.multiselect(
                "Manifest variables",
                options=[f"X{idx + 1}" for idx in range(st.session_state.DATA.shape[1])],
                help="Only these manifest variables will be considered in the diagnostics of the factor model."
            )
            number_of_factors = st.slider(label="Number of factors", step=1,
                                          value=3, min_value=1, max_value=(st.session_state.DATA.shape[1] - 1),
                                          help="The number of factors cannot exceed the number of manifest "
                                               "variables.", width="stretch")

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
                    cmap="Purples", axis=0, subset=["COMMUNALITY", "KMO_MSA"], vmin=0, vmax=1.0
                )

                with st.expander("Communalities and adequacies"):
                    st.dataframe(model_analysis_styled)

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
                        markers=True,
                        title="Eigenvalue per factor"
                    )
                    fig_scree.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Kaiser Criterion")
                    st.markdown(f"""
                    The total sum of eigenvalues is 
                    :blue-badge[{np.round(df_eigenvalues["Sum of Squared Loadings"].sum(), 4)}] out 
                    of the theoretical maximum of :blue-badge[{len(manifest_vars)}].
                    """)
                    st.plotly_chart(fig_scree, width="stretch")
                    st.space()

                with st.expander("Correlations"):
                    corr_mat = data_subset.corr()
                    fig_corr = px.imshow(
                        corr_mat,
                        text_auto="0.2f",
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        zmin=-1, zmax=1,
                        title="Subsetted correlation matrix"
                    )
                    fig_corr.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                    fig_corr.update_yaxes(tickmode="linear", dtick=1)
                    fig_corr.update_layout(
                        height=min(650, max(25 * st.session_state.DATA.shape[1], 300))
                    )
                    st.plotly_chart(fig_corr, width="stretch")
        elif diagnostics_sub_tab == "Raw data explorer":
            st.button("View quick summary", width="stretch", on_click=view_data_dialog)
            st.space("xxsmall")
            if st.session_state.STATEMENTS is not None and st.session_state.STATEMENTS_DF is not None:
                df_explorer = st.session_state.DATA.copy()
                df_explorer.columns = [
                    f"X{idx + 1} - " + str(st.session_state.STATEMENTS_DF[
                                               st.session_state.STATEMENTS_DF["Variable"] == col
                                               ]["Statement"].item())
                    for idx, col in enumerate(df_explorer.columns)
                ]
            else:
                df_explorer = st.session_state.DATA
            pyg_app = StreamlitRenderer(df_explorer)
            pyg_app.explorer()

with tab_dashboard:
    st.badge(":material/info: If your screen is not wide enough for the horizontal layout, "
             "consider temporarily hiding the *Menu* sidebar. You can also hide or show columns in tables.",
             color="blue")
    if st.session_state.DATA is None or len(st.session_state.FACTOR_MODELS) == 0:
        st.warning("Fit a factor model first.")
    else:
        col_1, col_2 = st.columns(2)
        with col_1:
            selected_models = st.multiselect(
                "Select models to examine", options=sorted(list(st.session_state.FACTOR_MODELS.keys())),
                help="You can choose up to 2 models at a time.",
                max_selections=2
            )

        st.space()
        if len(selected_models) == 0:
            st.warning("Choose at least one model.")
        else:
            multisets = [
                st.session_state.FACTOR_MODELS[model_name].generate_multiset(model_name)
                for model_name in selected_models
            ]
            model_analyses = [
                st.session_state.FACTOR_MODELS[model_name].analyze_model(model_name)
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

                    col_1, col_2, col_3 = st.columns(3)
                    col_1.metric("Variables", str(len(fit_details["manifest_vars"])))
                    col_2.metric("Factors", str(fit_details["number_of_factors"]))
                    v = st.session_state.FACTOR_MODELS[model_name].calculate_v_index(model_name)
                    v = np.round(v, 5) if v is not None else None
                    col_3.metric("V-Index", str(v))

                    col_4, col_5, col_6 = st.columns(3)
                    with col_4:
                        st.caption("ROTATION")
                        st.write(str(fit_details["rotation"]).capitalize())
                    with col_5:
                        st.caption("PRIOR TYPE")
                        st.write(str(fit_details["prior"]))

                    st.space()

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
                        cmap="Purples", axis=0, subset=["COMMUNALITY", "KMO_MSA"], vmin=0, vmax=1.0
                    )
                    st.dataframe(comm_and_adeq_styled, hide_index=True, key=f"{model_name}_comm_and_adeq")

                    st.space()

            # Interpretability plot
            cols = st.columns(len(selected_models), border=True)
            for i in range(len(selected_models)):
                with cols[i]:
                    model_name = selected_models[i]
                    multiset = multisets[i]

                    st.subheader(":material/psychology: Interpretability plot")
                    st.space()

                    similarity_type = ("Semantic Similarity"
                                       if st.session_state.FIT_DETAILS[model_name]["prior"] == "Semantics"
                                       else "Prior Similarity")

                    if multiset is None:
                        st.warning("No prior matrix was supplied for this model.")
                    else:
                        df_multiset = pd.DataFrame({
                            similarity_type: [item[0] for item in multiset],
                            "Loading Similarity": [item[1] for item in multiset]
                        })
                        lowess_failed = False
                        try:
                            with warnings.catch_warnings(record=True) as w:
                                fig_v_plot = px.scatter(
                                    df_multiset,
                                    x=similarity_type,
                                    y="Loading Similarity",
                                    trendline="lowess",
                                    title=f"{similarity_type} vs Loading Similarity",
                                    subtitle=f"V = "
                                             f"{st.session_state.FACTOR_MODELS[
                                                 model_name
                                             ].calculate_v_index(model_name)}"
                                )

                                if any("invalid value encountered in divide" in str(warn.message) for warn in w):
                                    raise RuntimeWarning("Lowess failed to fit.")
                        except RuntimeWarning:
                            fig_v_plot = px.scatter(
                                df_multiset,
                                x=similarity_type,
                                y="Loading Similarity",
                                trendline="ols",
                                title=f"{similarity_type} vs Loading Similarity",
                                subtitle=f"V = "
                                         f"{st.session_state.FACTOR_MODELS[model_name].calculate_v_index(model_name)}"
                            )
                            lowess_failed = True

                        fig_v_plot.update_layout(
                            yaxis=dict(
                                visible=True,
                                showticklabels=True,
                                showline=False,
                                showgrid=False,
                                zeroline=False
                            )
                        )
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
                        barmode="stack",
                        title="Sum of Squared Loadings per Factor"
                    )
                    fig_tags_breakdown.update_layout(
                        yaxis=dict(
                            visible=True,
                            showticklabels=False,
                            showline=False,
                            showgrid=False,
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

                    thresh = st.slider(
                        "Absolute threshold", min_value=0.0, max_value=1.0, value=0.35,
                        help="""
                        This discretizes the loadings such that a manifest variable is "included" in the 
                        factor (and its interpretation) if and only if the absolute value of the loading 
                        is at least the threshold.
                        """, key=f"{model_name}_thresh_slider"
                    )
                    threshs.append(thresh)
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
                        df_loadings = df_loadings_discretized

                    fig_loadings = px.imshow(
                        df_loadings,
                        text_auto="0.3f" if show_raw else "0",
                        aspect="auto",
                        color_continuous_scale="RdBu" if show_raw else "Purples",
                        color_continuous_midpoint=0 if show_raw else 0.5,
                        labels=dict(x="Factors", y="Variables",
                                    color="Loading" if show_raw else "Included"),
                        title="Factor Loading Matrix"
                    )
                    fig_loadings.update_xaxes(side="bottom", tickmode="linear", dtick=1)
                    fig_loadings.update_yaxes(tickmode="linear", dtick=1)
                    fig_loadings.update_layout(
                        height=min(900, max(50 * len(st.session_state.FIT_DETAILS[model_name]["manifest_vars"]), 300))
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
                            * Click on *Options* to see the legend and filters.
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
                    df_loadings_discretized[factor_cols] = df_loadings_discretized[factor_cols].abs().ge(
                        float(threshs[i])
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
                        col: df_loadings_discretized.index[df_loadings_discretized[col] == 1].tolist()
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
                            px.colors.qualitative.Plotly[factor]
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
                    df_loadings_discretized[factor_cols] = df_loadings_discretized[factor_cols].abs().ge(
                        float(threshs[i])
                    ).astype(int)

                    st.subheader(":material/cognition_2: Interpretation")
                    st.space()

                    if st.session_state.STATEMENTS is None:
                        st.warning("The associated statements for the variables were not provided.")
                    else:
                        st.markdown("Interpretations are generated using the groupings defined by the "
                                    "absolute threshold in :blue-badge[Factor loadings].")
                        st.caption("Note that this is not intended to replace the researcher's judgment and is "
                                   "only meant to help it. For instance, one should cross-check the LLM findings "
                                   "with the sign of the loadings, the cross-loadings, and so on.")
                        col_1, col_2 = st.columns(2)
                        with col_1:
                            if st.button("Generate interpretation",
                                         key=f"{model_name}_interpret_factor_model", width="stretch"):
                                st.session_state.INTERPRETATIONS[model_name] = interpret_factor_model(
                                    df_loadings_discretized
                                )
                        with col_2:
                            if st.button("Clear interpretation", width="stretch",
                                         key=f"{model_name}_clear_interpretation"):
                                st.session_state.INTERPRETATIONS[model_name] = (None, "")
                                st.rerun()
                        if st.session_state.INTERPRETATIONS[model_name][0] is not None:
                            st.space()
                            if st.session_state.INTERPRETATIONS[model_name][0] != "Error":
                                st.write(f"Generated by {st.session_state.INTERPRETATIONS[model_name][0]}")
                            st.write(st.session_state.INTERPRETATIONS[model_name][1])

                    st.space()

if show_floating_top:
    if floating_button(":material/keyboard_double_arrow_up: Top"):
        scroll_to_element(f"app_title_{st.session_state.SCROLL_COUNTER}")
