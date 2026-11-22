import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Document Clustering Tool", layout="wide")
st.title("📄 Document Clustering Tool")
st.caption("Group similar documents automatically using TF-IDF + KMeans. Upload .txt files, a .zip, or a CSV.")


# ---------- Data loading ----------
def load_documents():
    source = st.sidebar.radio("Data source", ["Upload .txt / .zip", "Upload CSV", "Sample data"])
    docs, names = [], []

    if source == "Upload .txt / .zip":
        files = st.sidebar.file_uploader(
            "Choose files", type=["txt", "zip"], accept_multiple_files=True
        )
        for f in files or []:
            if f.name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                    for n in z.namelist():
                        if n.endswith(".txt"):
                            docs.append(z.read(n).decode("utf-8", errors="ignore"))
                            names.append(n)
            else:
                docs.append(f.read().decode("utf-8", errors="ignore"))
                names.append(f.name)

    elif source == "Upload CSV":
        f = st.sidebar.file_uploader("CSV file", type=["csv"])
        if f:
            df = pd.read_csv(f)
            col = st.sidebar.selectbox("Text column", df.columns)
            docs = df[col].astype(str).tolist()
            names = [f"doc_{i}" for i in range(len(docs))]

    else:
        try:
            from sklearn.datasets import fetch_20newsgroups

            data = fetch_20newsgroups(
                subset="train",
                categories=["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast"],
                remove=("headers", "footers", "quotes"),
            )
            docs = data.data[:400]
            names = [f"news_{i}" for i in range(len(docs))]
        except Exception as e:
            st.error(f"Could not download the sample data (check your internet connection): {e}")

    # empty docs hata do
    pairs = [(n, d) for n, d in zip(names, docs) if d and d.strip()]
    return [p[0] for p in pairs], [p[1] for p in pairs]


names, docs = load_documents()

if len(docs) < 3:
    st.info("Load at least 3 documents from the sidebar to get started.")
    st.stop()

# ---------- Settings ----------
st.sidebar.header("Settings")
max_features = st.sidebar.slider("Max TF-IDF features", 500, 10000, 3000, 500)
ngram = st.sidebar.selectbox("N-grams", ["Unigrams", "Unigrams + Bigrams"])
k = st.sidebar.slider("Number of clusters (k)", 2, min(15, len(docs) - 1), 4)
viz = st.sidebar.radio("2D visualization", ["PCA", "t-SNE"])


# ---------- Pipeline ----------
@st.cache_data(show_spinner="Vectorizing...")
def vectorize(docs, max_features, ngram):
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2) if ngram != "Unigrams" else (1, 1),
        min_df=2 if len(docs) > 20 else 1,
        max_df=0.9,
    )
    X = vec.fit_transform(docs)
    return X, vec.get_feature_names_out()


@st.cache_data(show_spinner="Clustering...")
def cluster(_X, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(_X)
    return labels, km.cluster_centers_


@st.cache_data(show_spinner="Reducing dimensions...")
def reduce_2d(_X, method):
    if method == "PCA":
        Xd = TruncatedSVD(n_components=2, random_state=42).fit_transform(_X)
    else:
        from sklearn.manifold import TSNE

        Xs = TruncatedSVD(n_components=min(50, _X.shape[1] - 1), random_state=42).fit_transform(_X)
        Xd = TSNE(n_components=2, random_state=42, perplexity=min(30, len(docs) - 1)).fit_transform(Xs)
    return Xd


X, terms = vectorize(docs, max_features, ngram)
labels, centers = cluster(X, k)
coords = reduce_2d(X, viz)

df = pd.DataFrame(
    {
        "name": names,
        "cluster": labels.astype(str),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "preview": [d[:150].replace("\n", " ") + "..." for d in docs],
    }
)

# ---------- Output ----------
c1, c2, c3 = st.columns(3)
c1.metric("Documents", len(docs))
c2.metric("Features", X.shape[1])
try:
    c3.metric("Silhouette score", f"{silhouette_score(X, labels):.3f}")
except Exception:
    c3.metric("Silhouette score", "N/A")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Plot", "🔑 Top terms", "📑 Documents", "📈 Best k"])

with tab1:
    fig = px.scatter(
        df, x="x", y="y", color="cluster", hover_data=["name", "preview"],
        title=f"Document clusters ({viz})",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.bar_chart(df["cluster"].value_counts().sort_index())

with tab2:
    order = centers.argsort()[:, ::-1]
    for i in range(k):
        top = [terms[j] for j in order[i, :10]]
        st.markdown(f"**Cluster {i}** ({(labels == i).sum()} docs): " + ", ".join(top))

with tab3:
    sel = st.selectbox("Cluster filter", ["All"] + sorted(df["cluster"].unique()))
    view = df if sel == "All" else df[df["cluster"] == sel]
    st.dataframe(view[["name", "cluster", "preview"]], use_container_width=True)
    st.download_button(
        "⬇️ Download results as CSV", df[["name", "cluster"]].to_csv(index=False), "clusters.csv", "text/csv"
    )

with tab4:
    if st.button("Find the best k (elbow + silhouette)"):
        ks, inertias, sils = [], [], []
        for kk in range(2, min(11, len(docs))):
            km = KMeans(n_clusters=kk, n_init=5, random_state=42).fit(X)
            ks.append(kk)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, km.labels_))
        st.line_chart(pd.DataFrame({"inertia (elbow)": inertias}, index=ks))
        st.line_chart(pd.DataFrame({"silhouette": sils}, index=ks))
