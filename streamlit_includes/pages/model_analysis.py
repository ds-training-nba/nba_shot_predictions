import streamlit as st

from streamlit_includes.data.results import load_results


def render():
    df = load_results()
    st.markdown(
        '<div class="section-title">📉 Model Evaluation & Analysis</div>',
        unsafe_allow_html=True
    )
    tab1, tab2 = st.tabs(["Results and Metrics", "Error Analysis"])
    with tab1:
        col_left, col_right = st.columns([1, 1])

    with tab1,col_left:
        st.markdown(
            "## Metrics"
        )
        st.dataframe(df)
    with tab1,col_right:
        st.markdown(
            "## Conclusion"

        )
        st.markdown(
            " - Simple Lookup Table delivers comparable results to complex ML Models\n"
            " - Only ACTION_TYPE and PLAYER_NAME are required"
        )
    with tab2:
        col_left, col_right = st.columns([1, 1])
    with tab2, col_left:
        st.markdown(
            "## SHAP Analysis"
        )
        st.markdown(
            "An analysis of the local SHAP values of false predictions revealed the following:\n"
            " - rare ACTION_TYPE (shot technique) values are given too much importance\n"
            " - especially when taken appearing in a context with a sparse data support\n"
            " - supports the theory that the model relies almost exclusively on ACTION_TYPE and PLAYER_NAME"
        )
        st.markdown(
            "### Example"
        )
        st.markdown(
            "#### Details"
        )
        st.markdown(
            " - Fadeaway Bank Shot\n"
            " - By Dirk Nowitzki\n"
            " - From 16ft"
        )
        st.image(
            "doc/img/shap/action_type_overinterpretation.png",
            use_container_width=True
        )
        st.markdown(
            "#### Conclusion"
        )
        st.markdown(
            " - SHAP value for ACTION_TYPE shows much more importance than SHOT_DISTANCE\n"
            " - Not backed by physics!\n"
            " - Have a closer look at rare modalities and sparse appearance of a similar context\n"
            " - Maybe apply some _regularization_ to the model or _group_ the ACTION_TYPE modalities in question"
        )
    with tab2, col_right:
        st.markdown(
            "## Noise and Expectation Management"
        )

        st.markdown(
            "### The (perceived) Problem"
        )
        st.markdown(
            " - A lot of high confidence HIT predictions actually fail. \n"
            " - No shot of Dunk-ACTION_TYPE (and some other ACTION_TYPEs) is ever predicted as a miss (about 90% probability)"
            " - Looks like a bug (?)"
        )
        st.code(
            "...\n"
            "df_target_noise = df_full_dataset[\n"
            "(df_full_dataset['ACTION_TYPE'] == 'Running Dunk Shot') & \n"
            "(df_full_dataset['PLAYER_NAME'] == 'Kobe Bryant') & \n"
            "(df_full_dataset['SHOT_DISTANCE'] == 0)] \n"
            "\n"
            "sns.catplot(\n"
            "x='SHOT_MADE_FLAG', \n"
            "y='MINUTES_REMAINING', hue='scoreMarginBeforeShot', kind='bar', data=df_target_noise);"
        )
        st.image(
            "doc/img/error_analysis/noise.png",
            use_container_width=True
        )

        st.markdown(
            "### Question"
        )
        st.markdown(
            "How can the model deduce why the shots are missed?"
        )
        st.markdown(
            "### Answer"
        )
        st.markdown(
            "It can't. The explanatory variables (many of the more important ones) "
            "for hits and misses are pretty much the same for hits and misses. "
        )
        st.markdown(
            "### Conclusion"
        )
        st.markdown(
            "- inherent noise in the data, because of the domain\n"
            "- Shot outcome depends on non-existent or unmeasurable variables\n"
            "- focus on meaningful probabilities"
        )




