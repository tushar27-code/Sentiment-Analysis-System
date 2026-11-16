import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

# ---------------------------------
# 1. Load Dataset
# ---------------------------------

url = "https://docs.google.com/spreadsheets/d/1bjhOEAusrZydrRsnhOjwsCiBWkXJuZPQw3Q0S6HvCUc/export?format=csv&usp=sharing"

df = pd.read_csv(url)

# Check required column
if "Opinion" not in df.columns:
    raise ValueError("Dataset must contain an 'Opinion' column.")

print("\nOriginal Dataset:")
print(df.head())


# ---------------------------------
# 2. Sentiment Analysis using VADER
# ---------------------------------

model = SentimentIntensityAnalyzer()

sentiments = []
positive_scores = []
negative_scores = []
neutral_scores = []
compound_scores = []

for opinion in df["Opinion"]:

    # Handle missing or blank feedback
    if pd.isna(opinion) or str(opinion).strip() == "":
        scores = {
            "pos": 0,
            "neg": 0,
            "neu": 0,
            "compound": 0
        }
    else:
        scores = model.polarity_scores(str(opinion))

    positive_scores.append(scores["pos"])
    negative_scores.append(scores["neg"])
    neutral_scores.append(scores["neu"])
    compound_scores.append(scores["compound"])

    # Classify sentiment using standard VADER thresholds
    if scores["compound"] >= 0.05:
        sentiments.append("Positive")

    elif scores["compound"] <= -0.05:
        sentiments.append("Negative")

    else:
        sentiments.append("Neutral")


# Add results to dataset
df["Sentiment"] = sentiments
df["Positive Score"] = positive_scores
df["Negative Score"] = negative_scores
df["Neutral Score"] = neutral_scores
df["Compound Score"] = compound_scores


# ---------------------------------
# 3. Display Analysis Results
# ---------------------------------

print("\nSentiment Analysis Results:")
print(df.head(10))


# ---------------------------------
# 4. Calculate Sentiment Percentages
# ---------------------------------

total = len(df)

if total > 0:

    sentiment_counts = df["Sentiment"].value_counts()

    positive_count = sentiment_counts.get("Positive", 0)
    negative_count = sentiment_counts.get("Negative", 0)
    neutral_count = sentiment_counts.get("Neutral", 0)

    positive_percentage = (positive_count / total) * 100
    negative_percentage = (negative_count / total) * 100
    neutral_percentage = (neutral_count / total) * 100

    print("\nSentiment Summary:")
    print(f"Total Feedback: {total}")
    print(f"Positive: {positive_count} ({positive_percentage:.2f}%)")
    print(f"Negative: {negative_count} ({negative_percentage:.2f}%)")
    print(f"Neutral: {neutral_count} ({neutral_percentage:.2f}%)")


# ---------------------------------
# 5. Sentiment Pie Chart
# ---------------------------------

fig = px.pie(
    names=["Positive", "Negative", "Neutral"],
    values=[
        positive_count,
        negative_count,
        neutral_count
    ],
    title="Customer Feedback Sentiment Distribution",
    color_discrete_sequence=["green", "red", "gray"]
)

fig.show()

# ---------------------------------
# 6. Gender-wise Sentiment Analysis
# ---------------------------------

if "Gender" in df.columns:

    gender_fig = px.histogram(
        df,
        x="Gender",
        color="Sentiment",
        barmode="group",
        title="Gender-wise Customer Sentiment Analysis",
        labels={
            "Gender": "Gender",
            "count": "Number of Feedbacks"
        }
    )

    gender_fig.show()

else:
    print("\nGender column not found. Skipping gender analysis.")


# ---------------------------------
# 7. Export Results to CSV
# ---------------------------------

df.to_csv("results.csv", index=False)

print("\nAnalysis completed successfully!")
print("Results saved to results.csv")
