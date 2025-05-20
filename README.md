

Factor Correlation Structure (2000–Present)

Implementation Checklist & Deliverables

Goal: analyse the 2000-2025 evolution of correlations among ~90 monthly factor returns, identify persistent “super-clusters,” detect regime shifts, and present the findings with clear visualisations and interpretation.
Now with automated AI-uploadable summaries and visuals for downstream interpretation.

⸻

📁 Phase 0 — Project & Environment Setup

[ ] Create/activate Python env (≥3.10) with packages: pandas numpy scipy scikit-learn matplotlib networkx ruptures pyarrow fastparquet zarr python-louvain loguru scikit-bio
[ ] Clone / pull project repo and place raw data file
• Input: /data/T2_Optimizer.xlsx (Monthly_Net_Returns sheet)
[ ] Create /outputs and /reports folders
• Deliverable: folder structure committed to version control

🔧 Config, Logging & Dependency Lock

[ ] Create config.yml (window_length, corr_threshold, edge_threshold, random_seed, paths)
[ ] Initialise logs/ folder; integrate loguru with daily rotation
[ ] Generate locked env file (environment.lock.yml via conda-lock or pip-tools)

⸻

📁 Phase 0.5 — Data-Lake & Catalog

[ ] mkdir -p data/{raw,processed,derived,artifacts}
[ ] Convert raw Excel -> data/raw/factors_2000_2025.parquet (pyarrow)
[ ] pip install pyarrow fastparquet zarr python-louvain loguru
[ ] Init SQLite catalog.db with table assets
[ ] Write helper register_asset(path) that computes SHA-256, inserts row, aborts on dup

🔧 Config & Seeds

[ ] Create config.yml
window_length: 36
corr_threshold: 0.7
edge_threshold: 0.5
random_seed: 42

⸻

🧹 Phase 1 — Data Loading & Cleaning

[ ] Load Excel sheet; parse dates yyyy-mm-dd → index
[ ] Generate data-availability summary (first/last non-NA per factor)
[ ] Resolve missing values:
• Initial NA blocks → leave as NA (omit factor from window)
• Isolated gaps → (if rare) linear interpolate / ffill; log replacements
[ ] Winsorise extreme returns (>|50%|) if any, log changes
[ ] Export cleaned dataset
• Output 1: data/processed/returns_cleaned.parquet
• Output 2: reports/Data_Quality_Report.pdf

⸻

🔄 Phase 2 — Rolling 36-Month Correlation Matrices

[ ] Define rolling windows (2000-01 ➔ 2025-12)
[ ] For each window:
• Keep factors with full data in the 36-month slice
• Compute Pearson correlation matrix
[ ] Persist results (e.g., list of NumPy arrays)
• Output 3: data/derived/rolling_corr.zarr
[ ] Plot average pairwise correlation over time
• Output 4: reports/Average_Correlation_Timeline.pdf
[ ] Export summary CSV for AI upload (NEW)
• Output 4a: outputs/upload_bundle/avg_corr_series.csv (NEW)

⸻

🌳 Phase 3 — Per-Window Hierarchical Clustering

[ ] Convert correlation → distance d = 1 – ρ
[ ] Agglomerative linkage (average or complete)
[ ] Cut dendrogram at threshold τ ≈ 0.3 (ρ ≈ 0.7) → cluster labels
[ ] Store cluster labels per factor × window
• Output 5: data/derived/clusters/cluster_labels.parquet
• Output 5a: outputs/upload_bundle/cluster_labels.csv (NEW, wide or long format)
[ ] Dendrogram snapshots (e.g., 2003, 2008, 2020)
• Output 6: reports/Dendrogram_Snapshots.pdf

⸻

🔒 Phase 4 — Identify Persistent “Super-Clusters”

[ ] Build factor-pair co-occurrence matrix (% of windows in same cluster)
[ ] Cluster the co-occurrence matrix to define super-clusters
[ ] Document membership of each super-cluster
• Output 7: data/derived/superclusters.parquet
• Output 7a: outputs/upload_bundle/supercluster_stability.csv (NEW)
[ ] Co-occurrence heatmap
• Output 8: reports/Cooccurrence_Heatmap.pdf
• Output 8a: outputs/upload_bundle/cooccurrence_heatmap.png (NEW, for AI upload)

⸻

🔤 Phase 5 — Compare with Pre-Defined Factor Groups (optional if mapping exists)

[ ] Load / define mapping table Factor → Category
[ ] Compute within- vs across-group average correlations (rolling)
[ ] Adjusted Rand Index (ARI) between groups & clusters (rolling)
• Output 9: reports/Predefined_vs_Clusters.pdf

⸻

📈 Phase 6 — Principal Component Analysis

[ ] Overall PCA on full correlation matrix (continuous-factor subset)
[ ] Rolling eigen-decomposition per window
[ ] Track % variance of PC1, PC2; eigenvector similarity through time
• Output 10: reports/Eigenvalue_Trend.pdf
• Output 10a: outputs/upload_bundle/eigenvalues.csv (NEW)
• Output 11: reports/PC1_Similarity_Timeline.pdf
[ ] Scatter/biplot of factors in PC1-PC2 space (overall)
• Output 12: reports/PCA_Scatter.pdf
• Output 12a: outputs/upload_bundle/pca_scatter.pdf (NEW)

⸻

🔗 Phase 7 — Correlation Network & Community Detection

[ ] Select key windows (e.g., calm 2006-06; crisis 2008-10)
[ ] Construct network (edges if |ρ| > 0.5)
[ ] Run Louvain for communities; compute modularity & network metrics
[ ] Plot network graphs
• Output 13: reports/Network_Snapshots.pdf
[ ] Timeline of modularity score
• Output 14: reports/Modularity_Timeline.pdf
• Output 14a: outputs/upload_bundle/modularity.csv (NEW, modularity score per window)
• Output 14b: outputs/upload_bundle/modularity_timeline.pdf (NEW, for AI upload)

⸻

📊 Phase 8 — Structural Change & Regime Detection

[ ] Frobenius distance between successive correlation matrices
[ ] Apply change-point detection (ruptures)
[ ] Mantel tests & Procrustes comparisons for selected periods
• Output 15: data/derived/metrics/structural_change_metrics.parquet
[ ] Export changepoints
• Output 16: reports/Changepoint_Timeline.pdf
• Output 16a: outputs/upload_bundle/changepoints.csv (NEW, for AI upload)

⸻

🖼️ Phase 9 — Final Visualisation Suite

[ ] Rolling-heatmap montage (6-8 key dates)
[ ] Cluster-membership timeline heatmap
[ ] Combined figure: eigenvalue vs cluster-count timeline
• Output 17: reports/Final_Visualization_Package.pdf
[ ] Copy most important figures to upload folder (NEW)
• Output 17a: outputs/upload_bundle/average_corr_timeline.pdf (NEW)
• Output 17b: outputs/upload_bundle/final_viz.pdf (NEW, optional)

⸻

📝 Phase 10 — Report Compilation & Interpretation

[ ] Draft full analysis report (method, results, insights, limitations)
[ ] Add executive summary (1–2 pages)
• Output 18: reports/Factor_Correlation_Evolution_Report.pdf
• Output 19: reports/Executive_Summary.pdf
[ ] Auto-draft executive summary for AI upload (NEW)
• Output 19a: outputs/upload_bundle/executive_summary.txt (NEW, auto-generated)
[ ] Include AI prompt and file manifest
• Output 19b: outputs/upload_bundle/README_upload.txt (NEW)

⸻

✅ Phase 11 — Quality Assurance & Handover

[ ] Re-run key cells/scripts from scratch to confirm reproducibility
[ ] Cross-check a sample of numerical outputs vs manual sanity checks
[ ] Package code (scripts / notebooks) with environment file environment.yml
[ ] Tag repo release v1.0 and hand over to end users
• Deliverable: clean repository, reproducible environment, all outputs in /reports & /outputs
• Deliverable: All AI-ready outputs in outputs/upload_bundle/ (NEW)

⸻

AI Upload-Ready Files (summary)

All AI uploadable outputs (for ChatGPT-4o etc) are produced in:
outputs/upload_bundle/
	•	avg_corr_series.csv
	•	eigenvalues.csv
	•	modularity.csv
	•	changepoints.csv
	•	cluster_labels.csv
	•	supercluster_stability.csv
	•	cooccurrence_heatmap.png
	•	modularity_timeline.pdf
	•	average_corr_timeline.pdf
	•	pca_scatter.pdf
	•	executive_summary.txt
	•	README_upload.txt

Upload this folder to ChatGPT-4o and use the included prompt for auto-interpretation.

⸻

Version History

Date	Author	Notes
2025-06-20	AI draft	Initial checklist extracted from PRD
2025-06-21	AI update	Added explicit upload_bundle outputs, minimal structural changes