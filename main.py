import re
from collections import Counter
from datetime import datetime
from html import escape
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    h1 {
        color: #1e293b;
    }

    .block-container {
        padding-top: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# INITIALIZE VADER AND CONSTANTS
# ==========================================

mymodel = SentimentIntensityAnalyzer()

SENTIMENT_COLORS = {
    "Positive": "#16a34a",
    "Negative": "#dc2626",
    "Neutral": "#64748b"
}

SENTIMENT_ORDER = ["Positive", "Negative", "Neutral"]


# ==========================================
# HEADER
# ==========================================

st.title("📊 Sentiment Analysis System")
st.markdown("### Customers Feedback Analytics")
st.caption(
    "Analyze public opinions and understand product feedback."
)
st.divider()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("⚙️ Settings")
st.sidebar.info("Sentiment Analysis Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset",
        "Sentiment Analysis",
        "Advanced Analysis"
    ]
)


# ==========================================
# DATASET INPUT
# ==========================================

st.subheader("📁 Upload Dataset")

upload_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

sheet_url = st.text_input(
    "Or enter Google Sheets CSV URL",
    value=(
        "https://docs.google.com/spreadsheets/d/"
        "1bjhOEAusrZydrRsnhOjwsCiBWkXJuZPQw3Q0S6HvCUc/"
        "export?format=csv&usp=sharing"
    )
)


# ==========================================
# LOAD DATASET
# ==========================================

df = None

try:
    if upload_file is not None:
        df = pd.read_csv(upload_file)
    elif sheet_url.strip():
        df = pd.read_csv(sheet_url)

except Exception as e:
    st.error(f"Unable to load dataset: {e}")


# ==========================================
# SENTIMENT ANALYSIS FUNCTION
# ==========================================

def analyze_sentiment(text):
    """Return VADER sentiment label and individual scores."""

    if pd.isna(text) or not str(text).strip():
        return pd.Series({
            "Sentiment": "Neutral",
            "Positive Score": 0.0,
            "Negative Score": 0.0,
            "Neutral Score": 0.0,
            "Compound Score": 0.0
        })

    prediction = mymodel.polarity_scores(str(text))

    if prediction["compound"] >= 0.05:
        sentiment = "Positive"
    elif prediction["compound"] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return pd.Series({
        "Sentiment": sentiment,
        "Positive Score": prediction["pos"],
        "Negative Score": prediction["neg"],
        "Neutral Score": prediction["neu"],
        "Compound Score": prediction["compound"]
    })


def create_pdf_report(result_df):
    """Create a downloadable PDF summary of the sentiment analysis."""

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=45,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#1e3a8a"),
        alignment=TA_CENTER,
        spaceAfter=8
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=8
    )
    body_style = styles["BodyText"]
    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11
    )

    counts = result_df["Sentiment"].value_counts()
    total = len(result_df)
    positive = int(counts.get("Positive", 0))
    negative = int(counts.get("Negative", 0))
    neutral = int(counts.get("Neutral", 0))

    def percentage(value):
        return (value / total * 100) if total else 0

    story = [
        Paragraph("Customer Feedback Analytics", title_style),
        Paragraph("Sentiment Analysis Report", styles["Heading2"]),
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            body_style
        ),
        Spacer(1, 16),
        Paragraph("1. Executive Summary", section_style),
        Paragraph(
            f"This report summarizes <b>{total}</b> customer feedback records "
            "classified using the VADER Sentiment Intensity Analyzer.",
            body_style
        ),
        Spacer(1, 10)
    ]

    summary_data = [
        ["Sentiment", "Number of Opinions", "Percentage"],
        ["Positive", str(positive), f"{percentage(positive):.2f}%"],
        ["Negative", str(negative), f"{percentage(negative):.2f}%"],
        ["Neutral", str(neutral), f"{percentage(neutral):.2f}%"],
        ["Total", str(total), "100.00%" if total else "0.00%"]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[170, 170, 130],
        repeatRows=1
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")
    ]))
    story.extend([summary_table, Spacer(1, 14)])

    story.append(Paragraph("2. Key Findings", section_style))
    if total:
        if positive > negative and positive > neutral:
            leading_sentiment = "Positive"
        elif negative > positive and negative > neutral:
            leading_sentiment = "Negative"
        elif neutral > positive and neutral > negative:
            leading_sentiment = "Neutral"
        else:
            leading_sentiment = "There is no single leading sentiment"

        story.append(Paragraph(
            f"The most represented sentiment is <b>{leading_sentiment}</b>. "
            f"Positive feedback accounts for {percentage(positive):.2f}%, "
            f"negative feedback for {percentage(negative):.2f}%, and "
            f"neutral feedback for {percentage(neutral):.2f}% of all records.",
            body_style
        ))
    else:
        story.append(Paragraph("No feedback records were available.", body_style))

    negative_df = result_df[result_df["Sentiment"] == "Negative"].copy()
    if not negative_df.empty and "Opinion" in negative_df.columns:
        story.append(Paragraph("3. Sample Negative Feedback", section_style))
        feedback_data = [["No.", "Customer Opinion"]]
        for index, opinion in enumerate(negative_df["Opinion"].head(5), start=1):
            safe_opinion = escape(str(opinion))[:350]
            feedback_data.append([
                str(index),
                Paragraph(safe_opinion, small_style)
            ])

        feedback_table = Table(
            feedback_data,
            colWidths=[40, 430],
            repeatRows=1
        )
        feedback_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7)
        ]))
        story.append(feedback_table)

    story.extend([
        Spacer(1, 18),
        Paragraph("Methodology", section_style),
        Paragraph(
            "Sentiment labels are assigned using VADER compound scores: "
            "Positive (0.05 or above), Negative (-0.05 or below), and "
            "Neutral (between -0.05 and 0.05). This automated classification "
            "is an analytical aid and may not capture every context or nuance.",
            body_style
        )
    ])

    document.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def show_download_buttons(result_df, key_prefix):
    """Display CSV and PDF download buttons for analyzed feedback."""

    csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
    pdf_data = create_pdf_report(result_df)

    st.markdown("#### 📥 Download Analysis Report")
    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            label="⬇️ Download Results (CSV)",
            data=csv_data,
            file_name="customer_feedback_analysis.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv"
        )

    with download_col2:
        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_data,
            file_name="customer_feedback_report.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf"
        )


# ==========================================
# MAIN APPLICATION
# ==========================================

if df is None:
    st.warning(
        "Please upload a dataset or provide a valid Google Sheets URL."
    )

elif "Opinion" not in df.columns:
    st.error("Dataset must contain an 'Opinion' column.")

else:
    st.success("Dataset loaded successfully!")

    # Dataset information
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Opinions", df["Opinion"].isna().sum())

    st.divider()

    # ======================================
    # DATASET PAGE
    # ======================================

    if menu == "Dataset":
        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    # ======================================
    # SENTIMENT ANALYSIS PAGE
    # ======================================

    elif menu == "Sentiment Analysis":
        st.subheader("🧠 Sentiment Analysis Engine")

        st.write(
            "Analyze customer feedback using "
            "VADER Sentiment Intensity Analyzer."
        )

        if st.button(
            "🚀 Start Sentiment Analysis",
            type="primary"
        ):
            with st.spinner("Analyzing customer opinions..."):
                sentiment_results = df["Opinion"].apply(
                    analyze_sentiment
                )

                result_df = pd.concat(
                    [
                        df.reset_index(drop=True),
                        sentiment_results.reset_index(drop=True)
                    ],
                    axis=1
                )

                st.session_state["analyzed_df"] = result_df

            st.success("Sentiment analysis completed successfully!")

        if "analyzed_df" in st.session_state:
            result_df = st.session_state["analyzed_df"]

            st.divider()
            st.subheader("📊 Analysis Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "😊 Positive",
                (result_df["Sentiment"] == "Positive").sum()
            )
            col2.metric(
                "😞 Negative",
                (result_df["Sentiment"] == "Negative").sum()
            )
            col3.metric(
                "😐 Neutral",
                (result_df["Sentiment"] == "Neutral").sum()
            )

            st.divider()
            st.subheader("📋 Customer Sentiment Results")

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()
            show_download_buttons(result_df, "analysis_page")

    # ======================================
    # DASHBOARD PAGE
    # ======================================

    elif menu == "Dashboard":
        st.subheader("📈 Customer Feedback Dashboard")

        if "analyzed_df" not in st.session_state:
            st.info(
                "Please go to Sentiment Analysis and click "
                "'Start Sentiment Analysis' first."
            )

        else:
            result_df = st.session_state["analyzed_df"]

            sentiment_counts = (
                result_df["Sentiment"]
                .value_counts()
                .reindex(SENTIMENT_ORDER, fill_value=0)
            )

            total = len(result_df)
            positive = int(sentiment_counts["Positive"])
            negative = int(sentiment_counts["Negative"])
            neutral = int(sentiment_counts["Neutral"])

            positive_per = positive / total * 100 if total else 0
            negative_per = negative / total * 100 if total else 0
            neutral_per = neutral / total * 100 if total else 0

            # KPI cards
            st.markdown("#### 📌 Sentiment Overview")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Opinions", total)
            col2.metric("😊 Positive", positive, f"{positive_per:.1f}%")
            col3.metric("😞 Negative", negative, f"{negative_per:.1f}%")
            col4.metric("😐 Neutral", neutral, f"{neutral_per:.1f}%")

            st.divider()

            # Pie and bar charts
            st.markdown("#### 🍩 Sentiment Distribution")

            chart_col1, chart_col2 = st.columns(2)

            pie_df = pd.DataFrame({
                "Sentiment": sentiment_counts.index,
                "Count": sentiment_counts.values
            })

            with chart_col1:
                fig = px.pie(
                    pie_df,
                    names="Sentiment",
                    values="Count",
                    hole=0.5,
                    color="Sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                    title="Customer Sentiment Distribution"
                )
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(height=420, legend_title="Sentiment")

                st.plotly_chart(fig, use_container_width=True)

            with chart_col2:
                fig = px.bar(
                    pie_df,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                    text="Count",
                    title="Sentiment Comparison"
                )
                fig.update_layout(
                    height=420,
                    xaxis_title="Sentiment",
                    yaxis_title="Number of Opinions",
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Percentage chart
            st.markdown("#### 📊 Sentiment Percentage Analysis")

            percentage_df = pd.DataFrame({
                "Sentiment": SENTIMENT_ORDER,
                "Percentage": [
                    positive_per,
                    negative_per,
                    neutral_per
                ]
            })

            fig = px.bar(
                percentage_df,
                x="Sentiment",
                y="Percentage",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                text_auto=".1f",
                title="Percentage of Customer Opinions"
            )
            fig.update_layout(
                yaxis_title="Percentage (%)",
                xaxis_title="Sentiment",
                showlegend=False,
                yaxis_range=[0, 100]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Gender-wise analysis
            st.markdown("#### 👥 Gender-wise Sentiment Analysis")

            if "Gender" in result_df.columns:
                gender_df = result_df.copy()
                gender_df["Gender"] = (
                    gender_df["Gender"]
                    .fillna("Unknown")
                    .astype(str)
                )

                gender_counts = (
                    gender_df.groupby(["Gender", "Sentiment"])
                    .size()
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    gender_counts,
                    x="Gender",
                    y="Count",
                    color="Sentiment",
                    barmode="group",
                    color_discrete_map=SENTIMENT_COLORS,
                    text="Count",
                    title="Gender-wise Customer Sentiment"
                )
                fig.update_layout(
                    height=450,
                    xaxis_title="Gender",
                    yaxis_title="Number of Opinions"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Gender column not found. "
                    "Gender-wise analysis is unavailable."
                )

            st.divider()

            # Summary
            st.markdown("#### 📝 Sentiment Summary")

            st.write(
                f"😊 **{positive_per:.1f}%** of customer opinions are positive."
            )
            st.write(
                f"😞 **{negative_per:.1f}%** of customer opinions are negative."
            )
            st.write(
                f"😐 **{neutral_per:.1f}%** of customer opinions are neutral."
            )

            st.divider()
            show_download_buttons(result_df, "dashboard_page")

    # ======================================
    # ADVANCED ANALYSIS PAGE
    # ======================================

    elif menu == "Advanced Analysis":
        st.subheader("🔍 Advanced Customer Feedback Analysis")

        if "analyzed_df" not in st.session_state:
            st.warning("Please run Sentiment Analysis first.")

        else:
            result_df = st.session_state["analyzed_df"].copy()

            # Search
            st.markdown("#### 🔎 Search Customer Opinions")

            search_text = st.text_input(
                "Enter keyword to search",
                placeholder="Example: delivery, quality, price..."
            )

            # Sentiment filter
            st.markdown("#### 🎯 Filter by Sentiment")

            sentiment_filter = st.multiselect(
                "Select Sentiment",
                SENTIMENT_ORDER,
                default=SENTIMENT_ORDER
            )

            filtered_df = result_df[
                result_df["Sentiment"].isin(sentiment_filter)
            ].copy()

            if search_text.strip():
                filtered_df = filtered_df[
                    filtered_df["Opinion"]
                    .fillna("")
                    .astype(str)
                    .str.contains(
                        search_text,
                        case=False,
                        regex=False
                    )
                ]

            st.divider()

            # Filtered statistics
            col1, col2 = st.columns(2)

            col1.metric("Matching Opinions", len(filtered_df))
            col2.metric(
                "Negative Opinions",
                (filtered_df["Sentiment"] == "Negative").sum()
            )

            # Filtered records
            st.markdown("#### 📋 Filtered Customer Feedback")

            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )

            # Negative feedback analysis
            st.divider()
            st.markdown("#### 🚨 Negative Customer Feedback")

            negative_df = result_df[
                result_df["Sentiment"] == "Negative"
            ].copy()

            if not negative_df.empty:
                st.metric("Total Negative Opinions", len(negative_df))

                st.dataframe(
                    negative_df[
                        ["Opinion", "Sentiment", "Compound Score"]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                # Basic frequent-word analysis
                all_negative_text = " ".join(
                    negative_df["Opinion"]
                    .dropna()
                    .astype(str)
                    .str.lower()
                )

                words = re.findall(r"\b[a-z]{3,}\b", all_negative_text)

                stopwords = {
                    "the", "and", "for", "was", "are",
                    "this", "that", "with", "but", "not",
                    "very", "have", "has", "had", "they",
                    "their", "from", "you", "your", "its",
                    "our", "too", "can", "could", "would",
                    "been", "were", "there", "what", "when",
                    "which", "will", "about", "because"
                }

                filtered_words = [
                    word for word in words if word not in stopwords
                ]

                common_words = Counter(filtered_words).most_common(10)

                if common_words:
                    word_df = pd.DataFrame(
                        common_words,
                        columns=["Word", "Frequency"]
                    )

                    fig = px.bar(
                        word_df,
                        x="Frequency",
                        y="Word",
                        orientation="h",
                        title="Most Common Words in Negative Feedback",
                        text="Frequency"
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        height=450
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "There are no frequent words to display in "
                        "the negative feedback."
                    )

            else:
                st.success("No negative customer feedback found!")
