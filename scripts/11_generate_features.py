from sentence_transformers import SentenceTransformer, util
import re
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / 'input'
OUTPUT_DIR = ROOT_DIR / 'output' / 'generate_features'

def split_sentences(text):
    """
    Simple sentence splitter. Replace with nltk or spacy for higher accuracy.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def word_count(text):
    """
    Count the number of words in a text.
    """
    return len(text.split())

def character_count(text):
    """
    Count the number of characters in a text.
    """
    return len(text)

def sentence_count(text):
    """
    Count the number of sentences in a text.
    """
    return len(split_sentences(text))

def coherence_score(text, model=None, model_name="all-mpnet-base-v2"):
    """
    Computes semantic coherence between consecutive sentences.
    """

    model = model or SentenceTransformer(model_name)

    # Split into sentences
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return 0.0

    # Compute sentence embeddings
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # Compute cosine similarity between adjacent sentences
    similarities = [
        util.cos_sim(embeddings[i], embeddings[i + 1]).item()
        for i in range(len(embeddings) - 1)
    ]

    # Average coherence score
    return sum(similarities) / len(similarities)

def style_similarity_score(text, model=None, model_name="StyleDistance/styledistance"):
    """
    Computes stylistic similarity between consecutive sentences
    using StyleDistance embeddings.
    """
    model = model or SentenceTransformer(model_name)

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return 0.0

    embeddings = model.encode(sentences, normalize_embeddings=True)
    similarities = [
        util.cos_sim(embeddings[i], embeddings[i + 1]).item()
        for i in range(len(embeddings) - 1)
    ]
    return sum(similarities) / len(similarities)


if __name__ == "__main__":

    print("Loading ratings...")
    # Use combined majority-vote ratings; exclude invalid-majority
    combined = pd.read_csv(
        ROOT_DIR / 'ratings' / 'all_ratings.csv'
    )
    if 'excluded_invalid_majority' in combined.columns:
        combined = combined[~combined['excluded_invalid_majority']].copy()
    # Map final_label ('ai'/'human') to label ('Fraudulent'/'Authentic')
    if 'final_label' not in combined.columns:
        raise ValueError("ratings/all_ratings.csv must include 'final_label'")
    ratings = combined[['JournalUniqueID', 'final_label']].copy()
    ratings['label'] = ratings['final_label'].map({'ai': 'AI', 'human': 'Human'})
    ratings = ratings.drop(columns=['final_label'])

    # Load journals and filter to those present in ratings
    df = pd.read_csv(INPUT_DIR / "journals.csv")
    df = df[df["JournalUniqueID"].isin(ratings["JournalUniqueID"])].copy()
    df["JournalAnonymised"] = df["JournalAnonymised"].fillna("").astype(str)

    model = SentenceTransformer("all-mpnet-base-v2")
    style_model = SentenceTransformer("StyleDistance/styledistance")

    def _score(text: str) -> float:
        return coherence_score(text, model=model)

    print("Scoring coherence and style...")
    df_out = pd.DataFrame({
        "JournalUniqueID": df["JournalUniqueID"],
        "word_count": df["JournalAnonymised"].apply(word_count),
        "sentence_count": df["JournalAnonymised"].apply(sentence_count),
        "character_count": df["JournalAnonymised"].apply(character_count),
        "coherence": df["JournalAnonymised"].apply(_score),
        "style_similarity": df["JournalAnonymised"].apply(lambda t: style_similarity_score(t, model=style_model)),
    })

    # append manual review label from ratings
    df_out = df_out.merge(
        ratings[["JournalUniqueID", "label"]],
        on="JournalUniqueID",
        how="left",
    )
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_dir / "features.csv", index=False)
