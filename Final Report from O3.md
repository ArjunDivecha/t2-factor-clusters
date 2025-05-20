Factor-Correlation Structure Evolution

Monthly returns for ~90 investment factors, 2000 - 2025

⸻

1  Executive summary

Theme	What we found	Portfolio takeaway
Persistent “super-clusters”	Two hard blocs dominate the factor universe:  Macro-Momentum (commodity & rates-trend series) and  Equity Value-Quality (valuation & quality screens).  Members share the same cluster > 90 % of the time.	Treat each bloc as one risk unit; holding every member is redundancy, not diversification.
Correlation spikes in crises	Average pair-wise ρ jumps from a baseline 0.04-0.06 to ≈ 0.15 in Aug-2008 and Feb-2020, then retreats.	In stress, diversification evaporates.  Size risk to the crisis matrix, not the calm matrix.
Raised diversification “floor”	Post-GFC baseline correlations sit 30-40 % higher than in 2000-06; network modularity is lower on average.	Long-run risk models must use a higher base-ρ than early-2000s histories suggest.
Five structural break-points	2008-08, 2011-07, 2015-07, 2020-02, 2022-01 identified by Frobenius-distance spikes.	Use them to define six regimes and build regime-specific covariance sets.
Decoupling of Momentum after 2010	Momentum factors migrate out of the Value-Quality bloc and stay loosely correlated except in crises.	Combining Momentum with Value-Quality restores some cross-style diversification.


⸻

2  Data & preprocessing

Item	Detail
Source	T2_Optimizer.xlsx – Monthly_Net_Returns sheet (90 factors).
Horizon	Jan-2000 – Apr-2025 (304 months).
Cleaning	• Parsed to returns_cleaned.parquet  • Winsorised returns >
Storage	Tiered lake (Parquet / Zarr) + SQLite catalog; all analytics reproducible via run_all.py.
Upload bundle	AI-ready CSV / PDF / PNG files in outputs/upload_bundle/ (checked complete).


⸻

3  Methodology snapshot

Step	Technique	Key output
Rolling correlations	36-month window, stepped monthly	rolling_corr.zarr, avg_corr_series.csv
Hierarchical clustering	1 - ρ distance, τ = 0.3	cluster_labels.csv, dendrogram PDFs
Super-cluster detection	Factor-pair co-occurrence ≥ 80 %	supercluster_stability.csv, heat-map PNG
PCA	Eigen-decomp each window	eigenvalues.csv, scatter PDF
Network communities	Louvain on	ρ
Regime shifts	Normalised Frobenius distance; ruptures	changepoints.csv

All parameters live in config.yml; random seeds pinned (42).

⸻

4  Results

4.1  Correlation level through time

See average_corr_timeline.pdf
	•	Baseline ρ oscillates 0.03–0.06 until 2006.
	•	Spike 1: Aug-2008–Mar-2009 → avg ρ ≈ 0.15 (GFC).
	•	Spike 2: Feb-2020 → avg ρ ≈ 0.14 (COVID sell-off).
	•	Post-2010 “new normal”: floor settles ~0.08–0.10.

4.2  Cluster structure

Dendrogram snapshots & cooccurrence_heatmap.png

Super-cluster	Typical constituents	Persistence (windows)
Macro-Momentum	10-yr Bond_Carry CS/TS, Gold_TS, Oil_TS, G10 FX Trend	96 %
Equity Value-Quality	Earnings Yield_CS/TS, Debt-to-EV CS/TS, ROE_CS	92 %
Volatility/RSI mini-pack	20-Day Vol_CS/TS, RSI14_CS	78 %

CS/TS pairs lie in the same bloc > 98 % of the time → redundant exposures.

4.3  Principal components

See eigenvalues.csv & pca_scatter.pdf
	•	PC1 share averages 29 %, surging > 60 % during 2008 & 2020 shocks.
	•	PC1 loadings map almost one-for-one onto the Macro-Momentum bloc; PC2 contrasts Equity Value-Quality vs Momentum.
	•	Post-2010, Momentum loadings rotate away from Value, confirming cluster drift.

4.4  Network & modularity

See modularity_timeline.pdf
	•	Modularity averages 0.58 in calm periods → drops to 0.35 in crises.
	•	2015 commodity bust shows a local modularity dip within Macro-Momentum while equity blocs stay separate.

4.5  Detected regimes

changepoints.csv lists five statistically significant breaks (normalised Frobenius metric):

Date	Regime shift description
2008-08-31	GFC convergence – single common factor dominates.
2011-07-31	Euro-sovereign / US debt-ceiling stress – medium shock.
2015-07-31	China deval & commodity collapse – sector-specific.
2020-02-29	COVID panic – GFC-style but shorter.
2022-01-31	Inflation / rate-hike pivot – Value & commodities tighten, Momentum loosens.


⸻

5  Practical implications
	1.	Risk aggregation
	•	Treat each super-cluster as one macro factor in the risk model.
	•	Use crisis matrices (2008-09, 2020-03) as stress scenarios: expect 2.5× baseline portfolio variance.
	2.	Factor allocation
	•	Avoid double-owning CS & TS versions inside the same bloc.
	•	Pair Momentum with Value-Quality to recapture diversification lost post-2008.
	3.	Regime-aware sizing
	•	Scale gross when avg ρ > 0.12 or PC1 share > 50 %; diversification benefit is gone.
	4.	Forward monitoring
	•	Run the Frobenius metric live; alert if it breaches the 2022-01 level (~3 σ from mean).
	•	Keep an eye on modularity < 0.4 as early warning of cluster collapse.

⸻

6  Limitations & future work

Issue	Mitigation
CS/TS and similarly-named factors inflate bloc size	Consider collapsing to single representative series.
Pearson correlation only	Repeat key steps with Spearman to check robustness to fat tails.
NaN-driven changing universe	A second run restricted to factors live all 25 yrs would isolate survivorship effects.
Static 0.5 edge threshold in network	Parameter-sweep could reveal stability of community findings.


⸻

7  Appendices / artefacts

All raw tables, figures, and this report are reproducible via run_all.py.
Upload-ready artefacts live in outputs/upload_bundle/ for external AI narrative generation or committee decks.

⸻

Prepared for: Arjun – Investment Python
Date: 20 May 2025