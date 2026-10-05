import streamlit as st
import streamlit.components.v1 as components

# Google Analytics
GA_ID = "G-QXT3FWXQJE"

google_analytics = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){{dataLayer.push(arguments);}}
gtag('js', new Date());
gtag('config', '{GA_ID}');
</script>
"""

components.html(google_analytics, height=0)
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI College Predictor", layout="wide")

st.title("🎓 AI College Recommendation System")

# ===============================
# SESSION STATE
# ===============================
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []

if "trend_data" not in st.session_state:
    st.session_state.trend_data = None

if "results" not in st.session_state:
    st.session_state.results = None

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    pred_df = pd.read_csv("CollegeApp/data/xgboost_cutoff_predictions_2024.csv")
    original_df = pd.read_csv("CollegeApp/data/cleaned_tgeapcet_2017_2024.csv")

    le_college = pickle.load(open("CollegeApp/Model_pkl_files/le_college.pkl", "rb"))

    # Encode inst_code in original dataset
    original_df["inst_code"] = le_college.transform(original_df["inst_code"])

    # Map college names
    college_map = original_df[["inst_code", "institute_name"]].drop_duplicates()

    df = pred_df.merge(college_map, on="inst_code", how="left")

    return df, original_df

df, original_df = load_data()

# ===============================
# LOAD ENCODERS
# ===============================
@st.cache_resource
def load_encoders():
    le_branch = pickle.load(open("CollegeApp/Model_pkl_files/le_branch.pkl", "rb"))
    le_catgen = pickle.load(open("CollegeApp/Model_pkl_files/le_catgen.pkl", "rb"))
    return le_branch, le_catgen

le_branch, le_catgen = load_encoders()

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["📝 Predict", "📈 Trends", "⭐ Shortlist"])

# ===============================
# TAB 1: PREDICT
# ===============================
with tab1:

    st.header("Enter Your Details")

    col1, col2, col3, col4 = st.columns(4)

    rank = col1.number_input("Rank", min_value=1, step=1)

    category = col2.selectbox(
        "Category",
        ["OC", "BC_A", "BC_B", "BC_C", "BC_D", "BC_E", "SC", "ST"]
    )

    gender = col3.radio("Gender", ["Boy", "Girl"])

    branch = col4.selectbox("Branch", list(le_branch.classes_))

    # Filters
    st.subheader("Filters")

    f1, f2 = st.columns(2)

    chance_filter = f1.multiselect(
        "Chance",
        ["Safe", "High", "Moderate"],
        default=["Safe", "High", "Moderate"]
    )

    

    # Encode input
    branch_encoded = le_branch.transform([branch])[0]
    catgen = f"{category}_{'GIRLS' if gender == 'Girl' else 'BOYS'}"
    catgen_encoded = le_catgen.transform([catgen])[0]

    # ===============================
    # RECOMMEND FUNCTION
    # ===============================
    def recommend(df, rank, branch_encoded, catgen_encoded):

        data = df[
            (df["branch_code"] == branch_encoded) &
            (df["catgen_id"] == catgen_encoded)
        ].copy()

        if data.empty:
            return data

        data["score"] = (data["predicted_cutoff"] - rank) / rank

        def classify(score):
            if score >= 0:
                return "Safe"
            elif score >= -0.25:
                return "High"
            elif score >= -0.5:
                return "Moderate"
            else:
                return "Low"

        data["chance"] = data["score"].apply(classify)

        # Remove Low
        data = data[data["chance"] != "Low"]

        order = {"High": 1, "Moderate": 2, "Safe": 3}
        data["order"] = data["chance"].map(order)

        return data.sort_values(by=["order", "predicted_cutoff"])

    def get_color(chance):
        return {"Safe": "green", "High": "orange", "Moderate": "red"}[chance]

    # ===============================
    # PREDICT BUTTON
    # ===============================
    if st.button("🔍 Predict Colleges"):

        results = recommend(df, rank, branch_encoded, catgen_encoded)

        # Apply filters
        results = results[
            (results["chance"].isin(chance_filter)) 
            
        ]

        st.session_state.results = results

    # ===============================
    # SHOW RESULTS
    # ===============================
    if st.session_state.results is not None:

        results = st.session_state.results

        st.subheader("🎯 Recommended Colleges")

        if results.empty:
            st.error("No colleges found.")
        else:
            for i, (_, row) in enumerate(results.iterrows(), start=1):

                name = row["institute_name"]
                chance = row["chance"]
                color = get_color(chance)

                st.markdown(f"## #{i} 🎓 {name}")

                c1, c2, c3, c4 = st.columns(4)

                c1.write(f"📚 {branch}")
                c2.write(f"📊 Cutoff: {int(row['predicted_cutoff'])}")
                c3.markdown(
                    f"<span style='color:{color};font-weight:bold'>{chance}</span>",
                    unsafe_allow_html=True
                )

                # ⭐ Shortlist
                if c4.button("⭐", key=f"s{i}"):
                    st.session_state.shortlist.append(row)

                # 📈 Trend
                if c4.button("📈", key=f"t{i}"):
                    st.session_state.trend_data = row
                    st.success("👉 Go to Trends tab")

                st.divider()

# ===============================
# TAB 2: TRENDS
# ===============================
with tab2:

    st.header("📈 College Cutoff Trend (Last 5 Years)")

    if st.session_state.trend_data is not None:

        selected = st.session_state.trend_data

        inst = selected["inst_code"]
        branch_code = selected["branch_code"]

        # Decode branch
        branch_name = le_branch.inverse_transform([int(branch_code)])[0]

        # 🔥 FILTER FROM CLEANED DATASET
        trend_data = original_df[
            (original_df["inst_code"] == inst) &
            (original_df["branch_code"] == branch_name)
        ].copy()

        # 🔥 SORT YEARS
        trend_data = trend_data.sort_values("year")

        # 🔥 LAST 5 YEARS
        last_5_years = trend_data["year"].drop_duplicates().sort_values().tail(5)
        trend_data = trend_data[trend_data["year"].isin(last_5_years)]

        # 🔥 AGGREGATE
        trend_data = trend_data.groupby("year")["cutoff"].mean().reset_index()

        if not trend_data.empty:

            # ✅ BIGGER GRAPH
            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                trend_data["year"],
                trend_data["cutoff"],
                marker='o',
                linewidth=2
            )

            ax.set_xticks(trend_data["year"])

            ax.set_xlabel("Year")
            ax.set_ylabel("Cutoff Rank")
            ax.set_title(f"{selected['institute_name']} - {branch_name}")

            ax.grid(True)

            # ✅ REMOVE EXTRA BORDERS (clean look)
            ax.spines[['top', 'right']].set_visible(False)

            # ✅ VALUE LABELS
            for x, y in zip(trend_data["year"], trend_data["cutoff"]):
                ax.text(x, y, str(int(y)), fontsize=10, ha='center', va='bottom')

            plt.tight_layout()

            # 🔥 MOST IMPORTANT CHANGE
            st.pyplot(fig, use_container_width=True)

        else:
            st.warning("No trend data available.")

    else:
        st.info("👉 Select a college from Predict tab")
# ===============================
# TAB 3: SHORTLIST
# ===============================
with tab3:

    st.header("⭐ Shortlisted Colleges")

    if st.session_state.shortlist:

        shortlist_df = pd.DataFrame(st.session_state.shortlist)

        st.dataframe(
            shortlist_df[["institute_name", "predicted_cutoff", "chance"]]
        )

    else:
        st.info("No colleges shortlisted yet.")
