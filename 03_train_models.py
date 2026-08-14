import polars as pl
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import time
import joblib

def train_promarketa_intel():
    print("Initiating ProMarketa Machine Learning Engine...")
    start_time = time.time()

    # 1. INGESTION (Polars)
    print("Loading engineered digital marketing data...")
    df = pl.read_parquet("05_engineered_marketing_data.parquet")

    # 2. FEATURE SELECTION
    # Isolating core digital marketing and business metrics
    feature_cols = ["ctr", "roas", "ad_impressions", "ad_clicks", "campaign_cost", "historical_purchases"]
    
    # Drop nulls for safe ML training
    ml_df = df.drop_nulls(subset=feature_cols + ["churn_flag", "clv_proxy"])
    
    # Convert specifically to numeric Pandas arrays for universal ML compatibility
    X = ml_df.select(feature_cols).to_pandas()
    y_churn = ml_df.select(pl.col("churn_flag").cast(pl.Int32)).to_pandas().values.ravel()
    y_clv = ml_df.select(pl.col("clv_proxy").cast(pl.Float64)).to_pandas().values.ravel()

    # Split data to evaluate the Champion vs Challenger
    X_train, X_test, y_train, y_test = train_test_split(X, y_churn, test_size=0.2, random_state=42)

    # 3. CHAMPION / CHALLENGER ARENA (Churn Prediction)
    print("Initiating Champion vs. Challenger Training...")
    
    # Challenger A: LightGBM (Speed & Serverless Efficiency)
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
    lgb_score = roc_auc_score(y_test, lgb_preds)
    
    # Challenger B: XGBoost (The Industry Standard)
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    xgb_score = roc_auc_score(y_test, xgb_preds)

    print(f"LightGBM ROC-AUC: {lgb_score:.4f}")
    print(f"XGBoost ROC-AUC:  {xgb_score:.4f}")

    # 4. PROMOTE THE CHAMPION
    if lgb_score >= xgb_score:
        print("WINNER: LightGBM. Promoting to Production.")
        champion_model = lgb_model
    else:
        print("WINNER: XGBoost. Promoting to Production.")
        champion_model = xgb_model

    # Generate exact predictions for the ENTIRE dataset using the Champion
    final_churn_probs = champion_model.predict_proba(X)[:, 1]

    # 5. REGRESSION (Customer Lifetime Value)
    print("Training CLV Regression Engine...")
    clv_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clv_model.fit(X, y_clv)
    final_clv_preds = clv_model.predict(X)

    # 6. EXPLAINABLE AI (SHAP) PREPARATION
    # Verify the Champion model can be explained by SHAP before deploying to the dashboard
    print("Verifying Explainable AI (SHAP) compatibility...")
    explainer = shap.TreeExplainer(champion_model)
    _ = explainer.shap_values(X.head(5)) # Quick test to ensure dashboard won't crash

    # 7. INTEGRATE & SAVE
    print("Merging intelligence back into Polars Dataframe...")
    final_df = ml_df.with_columns([
        pl.Series("churn_probability", final_churn_probs).round(3),
        pl.Series("predicted_clv", final_clv_preds).round(2)
    ])
    
    # Save the ultimate dataset for Phase 4 (Streamlit)
    final_df.write_parquet("06_ProMarketa_intel_data.parquet")
    
    # Save the brains for SaaS API deployment
    joblib.dump(champion_model, "07_champion_churn_model.pkl")
    joblib.dump(clv_model, "08_clv_regressor_model.pkl")

    end_time = time.time()
    print(f"ML Pipeline executed successfully in {round(end_time - start_time, 2)} seconds.")
    print("Phase 3 Complete. Ready for Presentation Layer.")

if __name__ == "__main__":
    train_promarketa_intel()