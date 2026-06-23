import pandas as pd

# =========================
# Step 1: Load Dataset
# =========================

df = pd.read_csv("Dataset1.csv")

print("Original Dataset:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

# =========================
# Step 2: View Sample Reviews
# =========================

print("\nFirst Review:")
print(df["feedback"][0])

print("\nSecond Review:")
print(df["feedback"][1])

print("\nThird Review:")
print(df["feedback"][2])

# =========================
# Step 3: Split Dataset
# =========================

split_data = df["feedback"].str.split(",", expand=True)

print("\nSplit Dataset:")
print(split_data.head())

print("\nShape After Split:")
print(split_data.shape)

# =========================
# Step 4: Rename Columns
# =========================

split_data.columns = [
    "Review",
    "Sentiment",
    "Platform",
    "Date",
    "Username",
    "Location",
    "Score"
]

print("\nDataset After Renaming Columns:")
print(split_data.head())

# =========================
# Step 5: Missing Values Check
# =========================

print("\nMissing Values:")
print(split_data.isnull().sum())

# =========================
# Step 6: Dataset Information
# =========================

print("\nDataset Info:")
print(split_data.info())
# =========================
# Step 7: Sentiment Distribution
# =========================

print("\nSentiment Counts:")

sentiment_counts = split_data["Sentiment"].value_counts()

print(sentiment_counts)
# =========================
# Visualization 1
# Sentiment Distribution
# =========================

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))

sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.savefig(
    "charts/sentiment_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# =========================
# Visualization 2
# Sentiment Pie Chart
# =========================

plt.figure(figsize=(7,7))

plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct="%1.1f%%"
)

plt.title("Sentiment Percentage Distribution")

plt.savefig(
    "charts/sentiment_pie_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# =========================
# Visualization 3
# Reviews by Platform
# =========================

platform_counts = split_data["Platform"].value_counts()

print("\nPlatform Counts:")
print(platform_counts)
plt.figure(figsize=(10,5))

sns.barplot(
    x=platform_counts.index,
    y=platform_counts.values
)

plt.title("Reviews by Platform")
plt.xlabel("Platform")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=45)

plt.savefig(
    "charts/reviews_by_platform.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# step04
split_data["Score"] = pd.to_numeric(split_data["Score"])
platform_score = split_data.groupby(
    "Platform"
)["Score"].mean()

print("\nAverage Score by Platform:")
print(platform_score)
plt.figure(figsize=(10,5))

sns.barplot(
    x=platform_score.index,
    y=platform_score.values
)

plt.title("Average Sentiment Score by Platform")
plt.xlabel("Platform")
plt.ylabel("Average Score")

plt.xticks(rotation=45)

plt.savefig(
    "charts/average_score_by_platform.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# =========================
# Visualization 5
# Sentiment by Platform
# =========================

platform_sentiment = pd.crosstab(
    split_data["Platform"],
    split_data["Sentiment"]
)

print(platform_sentiment)
plt.figure(figsize=(10,6))

platform_sentiment.plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Sentiment by Platform")
plt.xlabel("Platform")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=45)

plt.savefig(
    "charts/sentiment_by_platform.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()




