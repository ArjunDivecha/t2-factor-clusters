⏺ Bash(cat ./outputs/upload_bundle/selected_factors.yml)…
  ⎿  - Best PE _CS
     - 120MA Signal_CS
     - 10Yr Bond_CS
     - 360 Day Vol_CS
     - 20 Day Vol_CS
     - 3MTR_CS
     - Inflation_CS
     - Currency Vol_CS
     - RSI14_CS
     - Shiller PE_CS
     - 3MTR_TS
     - Currency 12_CS
     - Inflation_TS
     - 120MA Signal_TS
     - 10Yr Bond_TS

⏺ Perfect! We've successfully implemented the factor selection analysis and added it to our pipeline. Here's what the factor selection process did:

  1. Selected anchor factors from the top 15 superclusters (one from each cluster)
  2. Screened for uniqueness to find factors with low collinearity to the anchors
  3. Applied Variance Inflation Factor (VIF) pruning to remove highly correlated factors
  4. Created a final list of 15 representative factors

  The selected factors include a diverse mix of:
  - Valuation factors: Best PE_CS, Shiller PE_CS
  - Technical indicators: 120MA Signal_CS/TS, RSI14_CS
  - Fixed income: 10Yr Bond_CS/TS
  - Volatility metrics: 360 Day Vol_CS, 20 Day Vol_CS
  - Momentum factors: 3MTR_CS/TS
  - Macro indicators: Inflation_CS/TS, Currency Vol_CS, Currency 12_CS

  This selection provides a more manageable subset of factors that still captures the essential dynamics of the factor space. The outputs are saved in both CSV and YAML
  formats in the upload_bundle directory, ready for use in downstream analysis or portfolio construction.

╭─────────