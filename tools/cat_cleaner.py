import os
import json
import pandas as pd
import numpy as np
import networkx as nx
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMerger:
    def __init__(
        self,
        file_path,
        threshold=1.5,
        comma_threshold=2,
        token_threshold=2,
        min_cluster_size=2,
        max_cluster_size=10,
        model_name='all-MiniLM-L6-v2'
    ):
        self.file_path = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]

        self.threshold = threshold
        self.comma_threshold = comma_threshold
        self.token_threshold = token_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

        self.df = pd.read_csv(file_path)
        self.model = SentenceTransformer(model_name)

        # Create json folder if not exists
        self.output_dir = "json"
        os.makedirs(self.output_dir, exist_ok=True)

    # -----------------------------
    # Utility Functions
    # -----------------------------
    def avg_comma_count(self, series):
        counts = series.dropna().astype(str).apply(lambda x: x.count(","))
        return counts.mean()

    def avg_token_length(self, series):
        tokens = series.dropna().astype(str).apply(lambda x: len(x.split()))
        return tokens.mean()

    def preprocess(self, series):
        raw_list = series.dropna().astype(str).unique().tolist()
        strings = [s.strip().lower() for s in raw_list if s.strip() != ""]
        return strings

    def has_numbers(self, s):
        return any(char.isdigit() for char in s)

    # -----------------------------
    # Graph-based Semantic Merge
    # -----------------------------
    def build_graph(self, strings):
        embeddings = self.model.encode(strings, convert_to_numpy=True)
        sim_matrix = cosine_similarity(embeddings)

        # Z-score normalization
        sim_mean = np.mean(sim_matrix)
        sim_std = np.std(sim_matrix)
        z_matrix = (sim_matrix - sim_mean) / sim_std

        G = nx.Graph()

        for s in strings:
            G.add_node(s)

        for i in range(len(strings)):
            for j in range(i + 1, len(strings)):
                if z_matrix[i, j] > self.threshold:
                    G.add_edge(strings[i], strings[j])

        return G

    def extract_clusters(self, G):
        clusters = []
        for component in nx.connected_components(G):
            size = len(component)

            if self.min_cluster_size <= size <= self.max_cluster_size:
                clusters.append(list(component))

        return clusters

    def merge_clusters(self, clusters):
        merged_output = {}

        for idx, cluster in enumerate(clusters):

            # Prevent merging numeric variants (your earlier issue)
            if any(self.has_numbers(s) for s in cluster):
                continue

            representative = min(cluster, key=len)

            merged_output[f"cluster_{idx}"] = {
                "original_strings": cluster,
                "merged_to": representative
            }

        return merged_output

    # -----------------------------
    # Main Runner
    # -----------------------------
    def run(self):
        final_output = {}

        for column in self.df.columns:
            series = self.df[column]

            comma_avg = self.avg_comma_count(series)
            token_avg = self.avg_token_length(series)

            # Filter columns (performance + quality)
            if comma_avg <= self.comma_threshold and token_avg <= self.token_threshold:

                strings = self.preprocess(series)

                if len(strings) < self.min_cluster_size:
                    continue

                G = self.build_graph(strings)
                clusters = self.extract_clusters(G)

                if clusters:
                    merged = self.merge_clusters(clusters)

                    if merged:
                        final_output[column] = merged

        # Save JSON
        output_path = os.path.join(self.output_dir, f"{self.file_name}_merged.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)

        print(f"Saved to: {output_path}")
        return final_output


class CatOutlierCleaner:
    """
    Combined pipeline: cosine-similarity clustering (testing3) → LLM merge suggestions.
    Output: {column: {cluster_id: {original_strings: [...], merged_to: "..."}}}
    """

    def __init__(
        self,
        file_path,
        llm,
        table_summary,
        threshold=0.75,
        dominant_min_pct=10,
        comma_threshold=20,
        token_threshold=50,
        min_cluster_size=2,
        max_cluster_size=20,
        model_name="all-MiniLM-L6-v2",
        output_dir="json",
    ):
        self.file_path = file_path
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.llm = llm
        self.table_summary = table_summary

        self.threshold = threshold
        self.dominant_min_pct = dominant_min_pct
        self.comma_threshold = comma_threshold
        self.token_threshold = token_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

        self.df = pd.read_csv(file_path)
        self.embed_model = SentenceTransformer(model_name)
        self._enc = tiktoken.encoding_for_model("gpt-4o-mini")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Column filter helpers
    # ------------------------------------------------------------------

    def _avg_comma(self, series):
        return series.dropna().astype(str).apply(lambda x: x.count(",")).mean()

    def _avg_tokens(self, series):
        vals = series.dropna().astype(str).str.strip().str.lower().unique()
        lengths = [len(self._enc.encode(v)) for v in vals]
        return float(np.mean(lengths)) if lengths else 0.0

    def _dominant_pct(self, series):
        vc = series.value_counts()
        total = series.dropna().shape[0]
        return (vc.iloc[0] / total * 100) if total > 0 else 0.0

    def _should_process(self, col):
        series = self.df[col]
        if self._dominant_pct(series) < self.dominant_min_pct:
            return False
        if self._avg_comma(series) > self.comma_threshold:
            return False
        if self._avg_tokens(series) > self.token_threshold:
            return False
        return True

    # ------------------------------------------------------------------
    # Cosine-similarity clustering (from testing3.ipynb)
    # ------------------------------------------------------------------

    def _cluster_column(self, col):
        series = self.df[col]
        raw = series.dropna().astype(str).unique().tolist()
        strings = [s.strip().lower() for s in raw if s.strip()]

        if len(strings) < self.min_cluster_size:
            return {}

        embeddings = self.embed_model.encode(strings, convert_to_numpy=True)
        sim_matrix = cosine_similarity(embeddings)

        G = nx.Graph()
        for s in strings:
            G.add_node(s)
        for i in range(len(strings)):
            for j in range(i + 1, len(strings)):
                if sim_matrix[i, j] >= self.threshold:
                    G.add_edge(strings[i], strings[j])

        cluster_dict = {}
        for idx, component in enumerate(nx.connected_components(G)):
            size = len(component)
            if self.min_cluster_size <= size <= self.max_cluster_size:
                cluster_dict[f"cluster_{idx}"] = sorted(list(component))

        return cluster_dict

    # ------------------------------------------------------------------
    # LLM merge suggestion
    # ------------------------------------------------------------------

    def _llm_merge(self, col, cluster_dict):
        """Call LLM and return parsed mergeable_clusters dict for this column."""
        payload = {col: cluster_dict}
        raw = self.llm.category_cleaner(
            column=col,
            table_summary=self.table_summary,
            database=json.dumps(payload, indent=2),
        )
        try:
            parsed = json.loads(raw)
            # LLM returns {"mergeable_clusters": {...}} — unwrap it
            return parsed.get("mergeable_clusters", parsed)
        except (json.JSONDecodeError, AttributeError):
            print(f"  [warn] could not parse LLM response for '{col}'")
            return {}

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(self, save=True):
        """
        Returns
        -------
        dict
            {column: {cluster_id: {original_strings: [...], merged_to: "..."}}}
        """
        final_output = {}

        cat_cols = self.df.select_dtypes(include=["object", "string"]).columns

        for col in cat_cols:
            if col == "ID":
                continue

            if not self._should_process(col):
                print(f"Skipping '{col}': failed column filter")
                continue

            print(f"Clustering '{col}' …")
            cluster_dict = self._cluster_column(col)

            if not cluster_dict:
                print(f"  No clusters found for '{col}'")
                continue

            print(f"  {len(cluster_dict)} clusters → calling LLM …")
            merged = self._llm_merge(col, cluster_dict)

            if merged:
                final_output[col] = merged

        if save:
            out_path = os.path.join(self.output_dir, f"{self.file_name}_llm_merged.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to: {out_path}")

        return final_output