
# Data Science Course by Sigma Software University
 Intensive 21-lesson program (10 weeks, 2 modules) focused on full-stack Data Science pipeline: from statistical modeling and machine learning to decision support and geospatial analysis.

## ️Table of Contents

- [🎯 Key Skills](#key-skills)
- [🧠 Final Projects](#final-projects-include)
- [🌐 Python Stack](#python-stack)
- [📜 Certificate](#certificate)
- [👤 About the Author](#about-the-author)
- [📄 License](#license)
- [📂 task_1 — Homework 1: Trend Modeling with Noise](#task_1--homework-1-trend-modeling-with-noise)
- [📂 task_2 — Homework 2: Real Data Modeling & Forecasting](#task_2--homework-2-real-data-modeling--forecasting)
- [📂 task_3 — Homework 3: Alpha-Beta(-Gamma) Filtering](#task_3--homework-3-alpha-beta-gamma-filtering-of-time-series )
- [📂 task_4 — Homework 4: Polynomial Regression & Forecasting](#task_4--homework-4-polynomial-regression--forecasting)
- [📂 task_5 — Homework 5: Drought-Resilient Crop Selection via SCOR Index](#task_5--homework-5-drought-resilient-crop-selection-via-scor-index)
- [📂 task_6 — Homework 6: ERP Analytics for Agro-Efficiency](#task_6--homework-6-erp-analytics-for-agro-efficiency)
- [📂 task_7 — Homework 7: Clustering Agro-Efficiency (SCOR-based DSS)](#task_7--homework-7-clustering-agro-efficiency-scor-based-dss)
- [📂 task_8 — Homework 8: Object Detection in Bathymetric Maps](#task_8--homework-8-object-detection-in-bathymetric-maps)
- [📂 task_9 — Homework 9: Customer Segmentation via Clustering](#task_9--homework-9-customer-segmentation-via-clustering)
- [📂 task_10 — Titanic Survival Prediction](#task_10--titanic-survival-prediction)
- [📂 task_11 — Sentiment Analysis, Text Mining & RNN (Spacy)](#task_11--sentiment-analysis-text-mining-and-rnn-spacy)
- [📂 task_12 — DEM Accuracy & ICESat-2 Ground Profile Extraction](#task_12--dem-accuracy-assessment-and-icesat-2-ground-profile-extraction)
---

## Key Skills
- Regression, Kalman filtering, anomaly detection
- DSS & ERP development with optimization (Google OR-Tools)
- Clustering, classification, neural networks (TensorFlow/Keras)
- Business forecasting, SCOR-index modeling, CRM credit scoring
- GIS analysis with GeoPandas, Digital Twins, e-commerce analytics


## Final projects include:
- DSS for drought-resilient crops
- Customer segmentation (KMeans, GMM, PCA)
- Neural network classifiers (Titanic)
- SCOR-based CRM simulator for credit risk
- GIS pipeline for terrain analysis with ICESat-2 data




## Python stack
`Numpy`, `Pandas`, `Statsmodels`, `Scikit-learn`, `Tensorflow`, `Keras`, `OpenCV`, `GeoPandas`, `Matplotlib`, `OR-Tools`, etc.



## Certificate

**✅ Data Science Certificate — Sigma Software University**  
🔗 [View Certificate](https://courses.university.sigma.software/certificates/0fdd0805968b4407aad6f8f2255284e5)

---



___

## task_1 — Homework 1: Trend Modeling with Noise


#### 📘 Description:

This folder contains code and scripts for Homework 1 (Level I), focused on:

* Modeling a **quadratic trend** with **normally distributed noise**
* Injecting **outliers**
* Analyzing **real meteorological temperature data** (via API or CSV)
* Applying **Least Squares Method (LSM)** for trend estimation

#### 🔧 Key Features:

* Synthetic data generation and noise simulation
* Polynomial trend fitting (2nd order)
* Residual analysis and histogram visualization
* Real data loading from API or local file

#### 🗃 Output:

* Graphs: trend, noise, outliers
* Residual statistics: mean, variance, standard deviation
* Example locations: **Kyiv, Kharkiv, Lviv**

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

--- 

## task_2 — Homework 2: Real Data Modeling & Forecasting

#### 📘 Description:

This folder contains scripts for analyzing real temperature data from meteorological stations and comparing forecasting models.

#### ✅ Key Features:

* Real-time data loading via API or synthetic generation
* Trend modeling using:

  * Polynomial regression (LSM)
  * sin-cos models
  * Exponential forecasting
  * Scikit-learn regressors (SGD, Ridge, etc.)
* Anomaly detection: medium, LSM, sliding window
* Statistical evaluation: R², residuals, cross-validation

#### 📊 Output:

* Visual trends and forecasts (+7 days)
* Comparison of model accuracy before and after outlier removal
* Real stations: Kyiv, Lviv, Kharkiv, etc.

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)


---

## task_3 — Homework 3: Alpha-Beta(-Gamma) Filtering of Time Series

#### 📘 Description:

This folder contains implementations of recurrent smoothing algorithms for time series data using:

* Alpha-Beta (α‑β) and Alpha-Beta-Gamma (α‑β‑γ) filters
* Kalman filters via FilterPy and PyKalman
* Both synthetic and real meteorological data

#### ✅ Key Features:

* Synthetic signal generation (quadratic + noise + anomalies)
* Real data loading from weather stations (via API)
* Outlier cleaning via sliding window method
* Filtering techniques:

  * Custom α‑β filter (adaptive and fixed variants)
  * Kalman filters (`FilterPy`, `PyKalman`)
* Forecasting (up to 6 future steps)

#### 📊 Results Summary:

* 📉 **Synthetic data (with noise and outliers):**
  α‑β filter after cleaning → R² ≈ **0.90**

* 🌡️ **Real temperature data (e.g., Kharkiv):**

  * α‑β (adaptive): R² ≈ **0.02**
  * α‑β (fixed α=0.4, β=0.1): R² ≈ **0.95**
  * FilterPy Kalman: R² ≈ **0.986**
  * PyKalman: R² ≈ **0.977**

* 🔮 Forecasts (6 hours ahead) show consistent short-term trends across methods.

#### 💡 Conclusion:

* **Fixed α‑β** performs better than adaptive α‑β on real data.
* **Kalman filters** (FilterPy, PyKalman) achieve superior smoothing and predictive accuracy.
* Proper preprocessing (outlier cleaning) significantly improves performance.

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---

## task_4 — Homework 4: Polynomial Regression & Forecasting

#### 📘 Description:

This folder contains a full pipeline for polynomial modeling and forecasting of temperature time series.

#### ✅ Key Features:

* Polynomial regression (degrees 1–4)
* Synthetic data with anomalies and noise
* Real station data (Kharkiv, Kyiv, etc.)
* Forecast extension: +0.5 step using regression
* Statistical evaluation:

  * Custom R², MSE
  * Residuals: mean, std, variance
* Optional anomaly removal via sliding window

#### 📊 Results:

* **Synthetic signal:**
  R² \~ 0.999, forecasting error ≈ 0.14

* **Real stations:**

  * Kharkiv: R² = 0.997
  * Kyiv: R² = 0.993
  * Forecasts within ±1°C accuracy

#### 💡 Conclusion:

* Polynomial regression (deg 2–3) performs best.
* Removing outliers before regression improves accuracy.
* Trend extension for +0.5 step is feasible for short-term forecasting.

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---

## task_5 — Homework 5: Drought-Resilient Crop Selection via SCOR Index


#### 📘 Description:

This folder contains a prototype of a Decision Support System (DSS) that evaluates drought-resilient crops using the SCOR (Soil-Crop-Outcome Resilience) index.

#### ✅ Key Features:

* **Input**: Agronomic dataset (yield, soil moisture, drought index)
* **Normalization**: Per-crop normalization of indicators
* **SCOR Calculation**: Weighted linear compromise between:

  * Yield
  * Soil moisture
  * Drought impact
* **Output**: Ranked crop suitability by region

#### 📊 Sample Results:

* **Top 3 crops** (lowest SCOR → more resilient):

  1. Soybean: 0.329
  2. Wheat: 0.331
  3. Barley: 0.347

* **Best crops for drought-affected regions**:

  * **Dnipropetrovska**: Wheat (0.74)
  * **Khersonska**: Soybean (0.71)
  * **Mykolaivska**: Soybean (0.87)
  * **Odeska**: Barley (0.77)
  * **Zaporizka**: Soybean (0.63)

#### 💡 Conclusion:

* The DSS model correctly identifies drought-resilient crops aligned with scientific literature (e.g., HB4 soybean, drought-resistant wheat strains).
* The system demonstrates high potential for real-world agri-decision support.

---
[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---

## task_6 — Homework 6: ERP Analytics for Agro-Efficiency


**Topic:** Building an integrated SCOR\_total index for crop decision support based on yield, fertilizer use, and cost.

### 🧠 Key Steps:

1. Merged datasets on yield, soil moisture, droughts, fertilizer use, and production costs.
2. Created a composite `SCOR_total` index using weighted criteria:

   * `SCOR_norm` (yield efficiency) – maximize
   * `Fert_SCOR` (fertilizer efficiency) – minimize
   * `Cost_SCOR` (economic cost) – minimize
3. Implemented dynamic weighting via a user interface.
4. Visualizations include top crops, heatmaps, OLAP 3D cube, and region-wise performance.

### 📊 Findings:

* **Top-performing crops:** Barley, Wheat, Soybean
* **Best regions (by SCOR\_total):** Volynska, Mykolaivska, Dnipropetrovska oblast
* **Tool purpose:** DSS-ready dashboard for optimizing crop selection by combining agronomic and economic indicators.

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---

## task_7 — Homework 7: Clustering Agro-Efficiency: SCOR-based DSS

**Topic:** Cluster analysis of agricultural efficiency using SCOR-based indicators and visual analytics.

---

### 🎯 Objective:

To develop a cluster analysis system for evaluating crop-region combinations based on yield, cost, and fertilizer use. Identify inefficient agro-decisions and visualize insights via heatmaps, dendrograms, and OLAP-style analytics.

---

### 🧩 Key Workflow:

1. Integrated agro-data: yield, moisture, droughts, fertilizer, and cost
2. Normalized key indicators: `SCOR_norm`, `Fert_SCOR`, `Cost_SCOR`
3. Computed weighted index: **SCOR\_total**
4. Performed KMeans clustering on:

   * Regions
   * Crops
   * Region × Crop pairs
5. Visualizations: scatter plots, heatmaps, dendrograms
6. Extracted insights: top crops per cluster, inefficient cases, regional recommendations

---

### 📊 Findings:

#### 🔹 Region Clusters:

* **Cluster 1 & 2:** High-performing regions (SCOR\_total ≈ 0.71–0.72)
* **Cluster 0:** Moderate performance (SCOR\_total ≈ 0.61)
* **Cluster 3:** Lowest efficiency (SCOR\_total ≈ 0.50)

#### 🔹 Top Crops by Cluster:

* **Cluster 1:** Barley, Potato, Maize
* **Cluster 2:** Wheat, Barley
* **Cluster 0:** Wheat, Sunflower
* **Cluster 3:** Potato, Soybean (low-performing)

#### 🔹 Regional Highlights:

* ✅ *Khersonska* (Sunflower) & *Volynska* (Barley) = High effectiveness
* ⚠️ *Vinnytska* & *Khmelnytska* = Underperforming across all crops

#### 🔥 Inefficient Crop-Region Pairs (SCOR < 0.55 & Cost > 0.2):

* Chernihivska – Maize (SCOR < 0.51, Cost = 1.0)
* Kharkivska – Sunflower, Wheat
* Sumska & Poltavska – multiple crops at risk

---

### 🧠 Conclusion:

The developed clustering model supports **data-driven agro-decision making** by clearly separating effective vs. inefficient crop-region combinations. The `SCOR_total` framework can be directly embedded into a DSS platform for agricultural planning and optimization under drought and cost constraints.

---
[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)


---

## task_8 — Homework 8: Object Detection in Bathymetric Maps

## 📌 Task Description

This assignment focuses on object detection in a bathymetric TIFF image using image processing techniques. The goal is to analyze terrain shapes or water body structures by detecting **corners** and **contours**.

## 🎯 Objectives

- Load and preprocess a bathymetric image
- Convert image to grayscale and negative
- Apply smoothing filters (Gaussian)
- Detect corners using Harris Detector
- Detect contours using Canny Edge Detection
- Count corners and contours
- Visualize the results

## 🛠️ Technologies Used

- Python 3.10+
- OpenCV
- NumPy
- Matplotlib

## 🧩 Main Features

- **Harris Corner Detection**: Identifies fine structural details.
- **Canny + Contours**: Highlights distinct object boundaries (e.g., terrain lines).
- **Visualization**: Compares both methods side-by-side with corner and contour counts.

## 🖼️ Sample Input

Bathymetric TIFF image: `map.tif`

## ✅ Results

| Method          | Description                        | Count         |
|-----------------|------------------------------------|---------------|
| Harris Corners  | High-resolution structural points  | ~314,000      |
| Canny Contours  | Detected isolated terrain shapes   | ~900+ contours|

## 📌 Conclusion

The **contour-based method** is more efficient for isoline extraction and shape analysis in bathymetric maps. Results can be applied to **cartography**, **terrain mapping**, and **hydrological analysis**.


---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---


## task_9 — Homework 9: Customer Segmentation via Clustering

#### 📘 Description:

This project explores customer segmentation using clustering techniques. The goal is to identify meaningful consumer groups based on demographic and behavioral data to support marketing strategy.

#### ✅ Key Features:

* Data cleaning: missing values, outlier removal (IQR)
* Feature engineering:

  * Age, Total Spending, Online Purchase Ratio
  * Education and Marital Status grouped into broader categories
* Data scaling (`StandardScaler`)
* Dimensionality reduction (`PCA`)
* Clustering algorithms:

  * K-Means
  * Agglomerative (Hierarchical)
  * DBSCAN
  * Gaussian Mixture
* Cluster evaluation metrics:

  * Silhouette Score
  * Calinski-Harabasz Index
  * Davies-Bouldin Score
* Visualizations:

  * Distribution plots, pairplots, scatter plots
  * 2D & 3D cluster projections
  * Dendrograms

#### 📊 Results:

* **Best clustering** (based on Silhouette Score):

  * K-Means with 3–4 clusters performed best
  * DBSCAN failed due to density settings
* PCA reduced dataset to 2–3 key components (explaining >60% variance)

#### 💡 Conclusion:

* Customer segmentation reveals distinct behavioral profiles.
* K-Means is effective for well-separated customer groups.
* Feature engineering (e.g., Average Spend, Parenthood) improved interpretability.

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---

## task_10 — Titanic Survival Prediction


#### 📘 Description:

Prediction of Titanic passenger survival using classical ML models and neural networks (Keras/TensorFlow).

#### ✅ Highlights:

* **Feature Engineering**: title extraction, ticket & cabin cleanup, encoding, scaling
* **ML Models**: Logistic Regression, SVM, Decision Tree (with `GridSearchCV`)
* **NN Tuning**: Layers, activation, dropout, optimizer, epochs
* **Metrics**: Accuracy, F1, Precision, ROC-AUC
* **Bonus**: CNN & LSTM on reshaped input

#### 📊 Best Result:

NN with `[33, 12, 8, 4]`, `ReLU`, `Adam`, `Dropout=0.2`
→ **Val Accuracy ≈ 83–85%**

#### 💡 Takeaway:

Feature engineering + tuned NN can outperform traditional models on tabular data.

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---

## task_11 — Sentiment Analysis, Text Mining and RNN (Spacy)

#### 📘 Description:

A full pipeline combining sentiment analysis using RNN (LSTM) on IMDB reviews, Ukrainian news scraping, and semantic similarity using spaCy word vectors.

---

#### ✅ Key Components:

* 🧠 **Sentiment Analysis with TensorFlow**

  * Dataset: IMDB reviews (`tensorflow_datasets`)
  * Model: `TextVectorization` + `Embedding` + `Bidirectional LSTM`
  * Evaluation: Accuracy & loss plots
  * Prediction on:

    * English text
    * Translated Ukrainian news (GoogleTranslator)

* 🌐 **News Parsing & Text Mining**

  * Sources: `rbc.ua`, `pressorg24.com`
  * Parsing via `requests` + `BeautifulSoup`
  * Saving news to `test_2.txt`
  * Text cleaning, tokenization, frequency analysis
  * Visualization: word clouds (`WordCloud`)

* 🧠 **Semantic Similarity with spaCy**

  * English & Ukrainian models (`en_core_web_sm`, `uk_core_news_sm`)
  * Cosine similarity of:

    * Sentences (document-to-document)
    * Words (token-to-token)
  * Applied to Ukrainian news vs. reference topics

---

#### 📊 Results:

* **Sentiment model** achieved \~85% validation accuracy (1 epoch)
* Word clouds & frequency stats reveal dominant themes in scraped news
* Semantic similarity detects meaningful overlaps in word usage

---

#### 💡 Conclusion:

* RNNs are effective for text classification with minimal preprocessing.
* Combined pipelines (scraping + translation + sentiment) enable multilingual analysis.
* spaCy’s word vectors capture contextual similarity even for Ukrainian text.

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---

## task_12 — DEM Accuracy Assessment and ICESat-2 Ground Profile Extraction

---

#### 📘 Description:

This project combines **accuracy evaluation of Digital Elevation Models (DEMs)** using **ICESat-2 ground truth points** with advanced **terrain profile extraction methods** from ATL03/ATL08 data.

---

### ✅ DEM Accuracy Analysis

**Input**: `icesat2_dems_with_deltas.parquet`
**Reference**: ICESat-2 ATL08 ground-classified points (`atl08_class == 1`, `atl03_cnf == 4`)

#### Metrics per DEM:

* MAE, RMSE, Bias, MedE, STD, IQR
* Bias trend interpretation ("overestimates"/"underestimates")

#### Visualizations:

* 📊 Bar + Line Plot: RMSE and MAE
* 📦 Boxplot of height errors vs. ICESat-2

**Best DEM**: FABDEM (MAE = 3.36 m, RMSE = 5.65 m)
**Worst DEM**: ASTER (RMSE = 10.84 m, IQR = 11.55 m)

---

### ✅ ICESat-2 Ground Profile Extraction

**Input**: `atl03x_*.parquet`
**Filtered by**: `rgt=998`, `cycle=1`, `spot=1`, `atl08_class==1`, `atl03_cnf==4`

#### 5 Methods:

1. **Rolling Minimum + Polynomial Fit**
2. **DBSCAN Clustering**
3. **Flexible Windowed Filtering**
4. **Sliding Local Regression (MNK)**
5. **Enhanced Local Regression with Shift & Tolerances**

#### Outputs:

* Cleaned ground profiles (`DataFrame`)
* Visualizations: white (raw), colored (cleaned by height)

---

### 📦 Output Files:

* `df_stat.csv` — DEM accuracy table
* `rmse_mae_plot.png`, `dem_error_boxplot.png` — visualization exports
* Cleaned ICESat-2 profiles ready for DEM comparison

---

### 💡 Conclusion:

* ICESat-2 is a robust reference for DEM quality control.
* Ground extraction quality significantly impacts DEM error statistics.
* Different cleaning methods suit different terrain and noise conditions.

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---

##  About the Author

**Viktor Nikoriak**  
PhD Researcher in Hydrology | Python Developer | GIS & ML Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/viktor-nikoriak-328404203/)  

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

[🔝 Back to top](https://github.com/NikoriakViktot/Data_Science_Course_SSWU#table-of-contents)

---


