import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class NumericalOutlierAnalyzer:
    """
    For each numeric column:
      1. Detects outliers via IQR * iqr_factor (default 1.5).
      2. Runs DBSCAN on the outlier values to find whether they cluster or are scattered noise.
      3. Reports: outlier %, mean, mode, IQR range, and DBSCAN cluster summary.
    """

    def __init__(self, df, iqr_factor=1.5, dbscan_min_samples=3, dbscan_eps=None):
        self.df = df
        self.iqr_factor = iqr_factor
        self.dbscan_min_samples = dbscan_min_samples
        # If eps is None it is auto-set per column as 0.5 * IQR of the outlier values
        self.dbscan_eps = dbscan_eps

    def _analyze_column(self, col):
        series = self.df[col].dropna()
        n = len(series)
        if n == 0:
            return None

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_factor * iqr
        upper = q3 + self.iqr_factor * iqr

        outliers = series[(series < lower) | (series > upper)]
        outlier_pct = len(outliers) / n * 100

        mean = series.mean()
        mode_vals = series.mode().tolist()
        mode = mode_vals[0] if mode_vals else None

        dbscan_summary = None
        if len(outliers) >= self.dbscan_min_samples:
            X = outliers.values.reshape(-1, 1)
            X_scaled = StandardScaler().fit_transform(X)

            eps = self.dbscan_eps
            if eps is None:
                outlier_iqr = np.percentile(outliers, 75) - np.percentile(outliers, 25)
                eps = max(0.3, 0.5 * outlier_iqr / (iqr if iqr > 0 else 1))

            labels = DBSCAN(eps=eps, min_samples=self.dbscan_min_samples).fit_predict(X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = int(np.sum(labels == -1))
            cluster_counts = {
                f"cluster_{i}": int(np.sum(labels == i))
                for i in set(labels) if i != -1
            }

            dbscan_summary = {
                "n_clusters": n_clusters,
                "noise_points": noise_count,
                "noise_pct": round(noise_count / len(outliers) * 100, 2),
                "cluster_counts": cluster_counts,
            }

        return {
            "mean": round(float(mean), 4),
            "mode": round(float(mode), 4) if mode is not None else None,
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "lower_fence": round(float(lower), 4),
            "upper_fence": round(float(upper), 4),
            "n_total": n,
            "n_outliers": len(outliers),
            "outlier_pct": round(outlier_pct, 2),
            "dbscan": dbscan_summary,
        }

    def run(self):
        """
        Returns
        -------
        dict
            {column: {mean, mode, q1, q3, iqr, lower_fence, upper_fence,
                       n_total, n_outliers, outlier_pct, dbscan: {...}}}
        """
        results = {}
        num_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            stats = self._analyze_column(col)
            if stats is not None:
                results[col] = stats
                print(
                    f"{col}: outlier_pct={stats['outlier_pct']}%  "
                    f"IQR=[{stats['q1']}, {stats['q3']}]  "
                    f"mean={stats['mean']}  mode={stats['mode']}  "
                    + (
                        f"dbscan_clusters={stats['dbscan']['n_clusters']}  "
                        f"noise={stats['dbscan']['noise_pct']}%"
                        if stats["dbscan"]
                        else "dbscan=too few outliers"
                    )
                )

        return results
