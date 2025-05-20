

Product Requirements Document:

Factor Correlation Structure Evolution Analysis (2000–Present)

⸻

Introduction & Background

Financial factors (or strategy returns) often exhibit complex interrelationships that can change over time. Understanding the correlation structure among a broad set of factors is crucial for portfolio diversification, risk management, and strategy development. This project focuses on analyzing how the correlations among approximately 90 monthly factor returns have evolved from the year 2000 to the present, using a rolling 3-year window. Prior research and practical observations suggest that while correlation patterns are time-varying, certain groupings of assets or strategies remain consistently related over long periods. For instance, in equity markets, after removing the market-wide factor, the remaining correlation structure often reveals stable sector or style groupings that persist through time. Likewise, some trading strategies show remarkably persistent correlations, indicating robust underlying common drivers.

By systematically examining rolling correlation matrices, we aim to uncover which relationships among these factors are persistent and form “superclusters” (i.e. groups of factors that stay highly correlated), and how/when the overall structure changes. The analysis will leverage various statistical techniques – from hierarchical clustering and principal component analysis (PCA) to network graphs and change-point detection – to provide a comprehensive view. The end result will be a thorough understanding of factor group dynamics, presented with clear visualizations and interpretation, and a plan that can be directly translated into code for implementation.

[ADDED/CHANGED:]
A secondary goal is to ensure all outputs (data tables, intermediate analytics, and visuals) are reproducible, compressed, catalogued, and—where relevant—directly uploadable as a “bundle” for downstream AI/LLM analysis (see “Deliverables”).

⸻

Objectives
	•	Evolution of Relationships: Examine how pairwise correlations between factors change over time, highlighting any significant shifts or trends in the correlation structure.
	•	Persistent Clusters (“Superclusters”): Identify groups of factors that consistently remain correlated with each other across time. These superclusters represent stable relationships (common risk or strategy themes) that persist despite market regime changes.
	•	Inclusion of Factor Returns Context: Incorporate insights from the factor return series themselves (e.g. their performance or volatility) to contextualize the correlation findings, ensuring interpretations consider both correlation and underlying return behavior.
	•	Predefined vs. Latent Groupings: Utilize any predefined factor groupings (if available, such as categories by asset class or strategy type) to compare against data-driven groupings. Also, explore latent groupings without preconceptions using techniques like PCA and clustering to see if natural structures emerge that might differ from predefined groups.
	•	Statistical Techniques: Apply a range of statistical and data analysis methods, including:
	•	Rolling correlation matrix computation (3-year trailing windows).
	•	Hierarchical clustering on correlation matrices to find clusters of factors per period.
	•	Principal Component Analysis (PCA) to detect major common components and latent factor groups.
	•	Network graph analysis of correlations to visualize and quantify factor network communities.
	•	Change-point detection methods (if applicable) to identify when significant changes in correlation structure occur.
	•	Quantitative comparison of correlation structures using measures like Frobenius norm differences, Procrustes analysis, and Mantel tests for matrix similarity.
	•	Visualization: Generate clear, insightful visualizations (rolling correlation heatmaps, network graphs, dendrograms, PCA plots, etc.) to illustrate how the factor correlation landscape evolves and to highlight persistent clusters or regime shifts.
	•	Interpretation Strategy: Develop a coherent interpretation of the results, linking statistical findings to practical insights (e.g. which factors constitute a single risk cluster, how diversification potential changes over time, and what external events correlate with structural changes). The analysis should ultimately guide how one might use these insights in portfolio construction or risk monitoring.

[ADDED:]
	•	Reproducibility & Output Structure: All analytics and figures must be produced from a reproducible pipeline, with clear intermediate and final outputs stored in a structured data lake and an AI-ready “upload bundle” directory (see below).

⸻

Data Description & Preprocessing

Data Overview

The dataset consists of ~90 monthly factor net returns spanning from 2000 to the present (2025). Each column represents a factor’s return time series, and each row corresponds to a month (with a Date field). Factors may include a variety of strategy styles and asset classes (e.g. equity styles like value or momentum, bond yield curve factors, technical indicators, macroeconomic strategy returns, etc.). Notably, many factors appear in pairs with suffixes like _CS and _TS, which likely denote related strategies (e.g. “Cross-Sectional” vs “Time-Series” implementations of a similar concept). We will treat each such series as a separate factor, but their relationship will be of interest (they might form obvious pairs or clusters if they are based on the same signal).

[ADDED/CHANGED:]
	•	File Format and Data Lake:
	•	Raw Excel (T2_Optimizer.xlsx) is converted to Parquet at ingest.
	•	Data is stored in a tiered folder structure:
	•	data/raw/ (immutable dumps, e.g., factors_2000_2025.parquet)
	•	data/processed/ (cleaned and aligned, e.g., returns_cleaned.parquet)
	•	data/derived/ (correlation matrices, cluster assignments, PCA results, etc.; using Parquet for tables, Zarr for N-D arrays)
	•	outputs/upload_bundle/ (see “Deliverables”)
	•	All large files are tracked in a simple SQLite or DuckDB catalog.db for provenance.

Preprocessing Steps
	•	Data Loading: Read the provided Excel file (Monthly_Net_Returns sheet of T2_Optimizer.xlsx) into a DataFrame, ensuring dates are properly parsed in chronological order.
	•	Handling Missing Values:
	•	Determine the start date for each factor (first non-missing entry).
	•	For each 36-month window, include only factors that have data for the entire window period.
	•	For sporadic gaps, interpolate or forward-fill only if absolutely needed for window continuity, otherwise omit that factor for the window.
	•	Consistency Checks: Ensure all returns are numeric and expressed in comparable terms. Winsorize extreme outliers (>|50%|).
	•	Predefined Group Info: If present, factor group mappings are loaded or inferred for later comparison.
	•	Data Span Selection: Start rolling windows only when enough factors are active to ensure matrix validity (e.g., skip windows with <5 factors).
	•	Output:
	•	Cleaned returns saved as data/processed/returns_cleaned.parquet.
	•	Data-availability and data-quality reports in reports/.

⸻

Rolling Window Correlation Analysis
	•	Window Definition:
	•	36-month rolling windows, step monthly.
	•	For each window, only factors with full data in the window are included.
	•	Correlation Matrix Calculation:
	•	Compute Pearson correlation matrix for each window.
	•	Store as chunked Zarr arrays (data/derived/rolling_corr.zarr), not as Pickle.
	•	Summary Metrics:
	•	Calculate average pairwise correlation for each window.
	•	[ADDED:] Export average correlations as CSV for AI upload: outputs/upload_bundle/avg_corr_series.csv.
	•	Skip and log windows with insufficient factors.

⸻

Hierarchical Clustering of Factors (per Window)
	•	Distance Metric:
	•	1 - ρ (correlation).
	•	Clustering Method:
	•	Agglomerative (average or complete linkage).
	•	Determining Clusters:
	•	Cut dendrogram at fixed threshold τ = 0.3 (ρ = 0.7).
	•	Number of clusters can vary by window.
	•	Cluster Labeling:
	•	Store per-factor, per-window cluster labels in Parquet (data/derived/clusters/cluster_labels.parquet).
	•	[ADDED:] Also output as CSV for AI upload (outputs/upload_bundle/cluster_labels.csv).
	•	Snapshots:
	•	Dendrograms for key years/periods (PDF in reports/).

⸻

Identification of “Superclusters” (Persistent Groups)
	•	Co-occurrence Matrix:
	•	Calculate the % of time each factor-pair clusters together.
	•	Cluster the co-occurrence matrix to define persistent “superclusters.”
	•	Outputs:
	•	Supercluster assignments stored as Parquet (data/derived/superclusters.parquet).
	•	[ADDED:] AI-ready CSV: outputs/upload_bundle/supercluster_stability.csv.
	•	Visual co-occurrence heatmap as PDF for report, PNG for AI upload:
reports/Cooccurrence_Heatmap.pdf,
outputs/upload_bundle/cooccurrence_heatmap.png.

⸻

Predefined vs. Latent Groupings (optional if mapping exists)
	•	Mapping table: Factor → Category loaded or inferred.
	•	Comparisons:
	•	Compute within/across group average correlations.
	•	ARI (Adjusted Rand Index) between predefined and found clusters per window.
	•	Output: Comparison plots as PDF (reports/Predefined_vs_Clusters.pdf).

⸻

Principal Component Analysis
	•	Overall PCA on the entire cleaned correlation matrix.
	•	Rolling PCA by window (eigen-decomposition of each window’s corr matrix).
	•	Track:
	•	% variance explained by PC1, PC2, etc.
	•	Eigenvector similarity over time.
	•	Outputs:
	•	Eigenvalues: data/derived/pca/eigenvalues.parquet
	•	Eigenvectors: data/derived/pca/eigenvectors.zarr
	•	[ADDED:] Export outputs/upload_bundle/eigenvalues.csv and outputs/upload_bundle/pca_scatter.pdf for AI review.
	•	PDF plots for human review:
reports/Eigenvalue_Trend.pdf, reports/PC1_Similarity_Timeline.pdf, reports/PCA_Scatter.pdf.

⸻

Correlation Network Analysis
	•	Network Construction:
	•	Nodes = factors; edges = |ρ| > threshold (e.g., 0.5).
	•	Community Detection:
	•	Louvain method; modularity tracked by window.
	•	Pin random seed for reproducibility.
	•	Outputs:
	•	Community labels and modularity:
data/derived/metrics/modularity.parquet
	•	[ADDED:] Export: outputs/upload_bundle/modularity.csv, outputs/upload_bundle/modularity_timeline.pdf.
	•	Network graph visuals: reports/Network_Snapshots.pdf,
reports/Modularity_Timeline.pdf.

⸻

Comparing Correlation Structures Over Time (Statistical Distance Measures)
	•	Frobenius Norm:
	•	Calculate window-to-window and regime-to-regime distances.
	•	Procrustes Analysis:
	•	Compare window pairs’ factor geometry.
	•	Mantel Test:
	•	Correlation between distance matrices for different periods.
	•	Eigenvector Similarity:
	•	Compute dot products between leading eigenvectors across time.
	•	Outputs:
	•	All metric time series in Parquet (data/derived/metrics/structural_change_metrics.parquet).
	•	[ADDED:] AI uploadable changepoint list:
outputs/upload_bundle/changepoints.csv
	•	PDF: reports/Changepoint_Timeline.pdf.

⸻

Change-Point Detection & Regime Analysis
	•	Apply change-point detection (ruptures or threshold-based) to:
	•	Frobenius distances
	•	First eigenvalue
	•	Modularity
	•	Annotate timelines and heatmaps with detected regimes.

⸻

Visualization Plan
	•	Rolling heatmaps (select windows)
	•	Dendrograms (selected periods)
	•	Cluster-membership timeline (heatmap)
	•	Co-occurrence heatmap (superclusters)
	•	PCA scatter and eigenvalue trend plots
	•	Network graphs (select periods)
	•	Distance/metric timelines and change-point annotations
	•	[ADDED:]
	•	All summary visuals, metrics, and timelines are duplicated as CSV/PDF/PNG in outputs/upload_bundle/ for AI upload.
	•	AI bundle file manifest and interpretation prompt included as README_upload.txt in the bundle.

⸻

Interpretation & Analysis Strategy

[Unchanged—see original for full breakdown of supercluster descriptions, temporal changes, regime explanations, implications for portfolio construction, etc.]

⸻

Implementation Plan & Technical Details
	•	Python ecosystem:
	•	pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, networkx, python-louvain, ruptures, scikit-bio, pyarrow, zarr, loguru.
	•	Config:
	•	All thresholds, window sizes, seeds, and paths set in config.yml.
	•	Data Lake Structure:
	•	Raw, processed, derived, artifacts (see above).
	•	SQLite catalog.db tracks all output paths and hashes.
	•	All intermediate analytics and final results stored in Parquet or Zarr.
	•	All AI-uploadable outputs produced automatically at the end of the pipeline.
	•	Testing:
	•	At least one toy fixture to run full pipeline in <10s for CI.
	•	Random seeds fixed for reproducibility.
	•	Logging:
	•	loguru writes to logs/ with daily rotation.
	•	File Overwrite Protection:
	•	Catalog.db prevents silent overwrites; only allow with explicit flag.

⸻

Deliverables
	•	Analysis Report: Full PDF covering intro, methodology, all key findings, visualizations, and interpretation.
	•	Code: Modular scripts/notebooks; reproducible from run_all.py.
	•	Cleaned Data:
	•	Parquet/Zarr format in structured data lake.
	•	All figures/visuals:
	•	High-res PNG/PDF in reports/ and AI-ready versions in outputs/upload_bundle/.
	•	AI/LLM Upload Bundle:
	•	CSV: avg_corr_series.csv, eigenvalues.csv, modularity.csv, changepoints.csv, cluster_labels.csv, supercluster_stability.csv
	•	PDF/PNG: cooccurrence_heatmap.png, modularity_timeline.pdf, average_corr_timeline.pdf, pca_scatter.pdf
	•	TXT: executive_summary.txt, README_upload.txt
	•	All under outputs/upload_bundle/
	•	Manifest and Metadata:
	•	All data products tracked in catalog.db and a YAML manifest for reproducibility.

⸻

Data Structure

factor_corr_project/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── derived/
│   └── artifacts/
├── outputs/
│   └── upload_bundle/
├── reports/
├── logs/
├── catalog.db
├── config.yml
├── environment.yml / environment.lock.yml
├── manifests/
└── scripts/


⸻

Risks & Gaps (and Fixes)

Gap	Fix
Data sprawl/output	Structured data lake, no Excel or Pickle for analysis
Output overwrite	Catalog.db logs all outputs, block overwrite unless forced
Parameter drift	All tunables in config.yml
Reproducibility	environment.lock.yml pinned; toy fixture for pipeline test
Randomness	All random seeds pinned in config and scripts
Manual intervention	All outputs (tables, figures, summaries) generated by code
AI integration	AI upload bundle created as last pipeline step, with manifest


⸻

In summary:
All pipeline steps, data storage, and outputs are fully reproducible, compressible, and catalogued.
Every important analytic or figure has both human-readable (PDF/PNG) and AI-ingestible (CSV, PNG, TXT) versions, so you can hand the entire outputs/upload_bundle directory to any LLM (such as ChatGPT-4o) and receive a complete narrative or executive summary.
No step or output is left untracked or ambiguous.

