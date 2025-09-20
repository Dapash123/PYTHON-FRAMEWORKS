# Part 1: Data Loading and Basic Exploration

import pandas as pd

# Load dataset (adjust path if you saved it elsewhere)
df = pd.read_csv(r"C:\Users\lenovo\Desktop\metadata.csv\metadata.csv", low_memory=False, nrows=5000)

# Preview first few rows
print("First 5 rows of dataset:")
print(df.head())

# Check data structure
print("\nDataset Info:")
print(df.info())

# Dimensions (rows, columns)
print("\nDataset Dimensions:", df.shape)

# Check missing values
print("\nMissing Values per Column:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

# Basic statistics
print("\nSummary Statistics (numerical columns):")
print(df.describe())


#Part 2: Data Cleaning and Preparation
# Handle missing values
missing_percent = df.isnull().mean() * 100
print("\nColumns with more than 50% missing values:")
print(missing_percent[missing_percent > 50])

# Example: drop highly missing columns
df_clean = df.drop(columns=missing_percent[missing_percent > 50].index)

# Fill missing titles/journals with "Unknown"
df_clean["title"] = df_clean["title"].fillna("Unknown Title")
df_clean["journal"] = df_clean["journal"].fillna("Unknown Journal")

# Convert publish_time to datetime
df_clean["publish_time"] = pd.to_datetime(df_clean["publish_time"], errors="coerce")

# Extract year
df_clean["year"] = df_clean["publish_time"].dt.year

# New feature: abstract word count
df_clean["abstract_word_count"] = df_clean["abstract"].dropna().apply(lambda x: len(str(x).split()))

print("\nCleaned Dataset Sample:")
print(df_clean.head()) 

# Part 3: Data Analysis and Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# ====================
# Papers by year
# ====================
papers_by_year = df_clean["year"].value_counts().sort_index()

plt.figure(figsize=(10,5))
sns.barplot(x=papers_by_year.index, y=papers_by_year.values, color="skyblue")
plt.title("Papers Published by Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.xticks(rotation=45)
plt.show()

# ====================
# Top Journals
# ====================
top_journals = df_clean["journal"].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(y=top_journals.index, x=top_journals.values, palette="viridis")
plt.title("Top 10 Journals Publishing COVID-19 Papers")
plt.xlabel("Number of Papers")
plt.ylabel("Journal")
plt.show()

# ====================
# Word Frequency in Titles
# ====================
all_words = " ".join(df_clean["title"].dropna().astype(str)).lower().split()
common_words = Counter(all_words).most_common(20)

print("\nTop 20 Most Common Words in Titles:")
for word, count in common_words:
    print(f"{word}: {count}")

# ====================
# Word Cloud
# ====================
wordcloud = WordCloud(width=1000, height=500, background_color="white", colormap="plasma").generate(" ".join(all_words))

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Paper Titles")
plt.show()

# ====================
# Distribution by Source
# ====================
top_sources = df_clean["source_x"].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(y=top_sources.index, x=top_sources.values, palette="coolwarm")
plt.title("Top Sources of Publications")
plt.xlabel("Number of Papers")
plt.ylabel("Source")
plt.show()