import polars as pl
import time

def engineering_features():
    print("Initializing Polars Lazy Compute Engine for 1,000,000 rows...")
    start_time = time.time()
    
    # 1. Pointing to the massive Parquet file without loading it into RAM
    lazy_df = pl.scan_parquet("04_raw_marketing_data.parquet")
    
    # 2. CLEAN & STANDARDIZE
    cleaning_plan = lazy_df.with_columns([
        # Safely handling any strings with commas in the purchase column
        pl.col("purchase_value")
            .cast(pl.String)
            .str.replace_all(",", "")
            .cast(pl.Float64),
            
        # Standardizing location strings
        pl.col("location")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_titlecase()
    ])
    
    # 3. ADVANCED METRICS (Method Chaining)
    feature_plan = (
        cleaning_plan.with_columns([
            
            # Feature 1: Click-Through Rate (CTR)
            # Logic: Avoid division by zero for Organic/Direct traffic
            pl.when(pl.col("ad_impressions") == 0)
                .then(0.0)
                .otherwise(pl.col("ad_clicks") / pl.col("ad_impressions"))
                .alias("ctr"),
                
            # Feature 2: Return on Ad Spend (ROAS)
            # Logic: Revenue generated divided by the cost to acquire the customer
            pl.when(pl.col("campaign_cost") == 0)
                .then(0.0)
                .otherwise(pl.col("purchase_value") / pl.col("campaign_cost"))
                .alias("roas"),
                
            # Feature 3: Customer Lifetime Value (CLV Proxy)
            # Logic: Their current purchase plus historical track record
            ((pl.col("historical_purchases") + 1) * pl.col("purchase_value"))
                .alias("clv_proxy")
                
        ])
        .with_columns([
            
            # Feature 4: Actionable Business Segmentation
            # Using conditional chaining to group 1,000,000 rows into precise marketing targets
            pl.when((pl.col("roas") > 2.5) & (pl.col("clv_proxy") > 500))
                .then(pl.lit("High-Value VIP"))
            .when(pl.col("cart_abandonment") == True)
                .then(pl.lit("High-Intent Abandoner (Retarget)"))
            .when((pl.col("churn_flag") == 1) | (pl.col("csat_score") < 4))
                .then(pl.lit("High Churn Risk (Win-back)"))
            .otherwise(pl.lit("Standard Viewer"))
                .alias("customer_segment")
                
        ])
    )
    
    # 4. EXECUTE: Trigger the Rust-based multithreaded engine
    print("Executing 1 Million Rows across all CPU cores...")
    final_df = feature_plan.collect()
    
    # 5. SAVE
    file_name = "05_engineered_marketing_data.parquet"
    final_df.write_parquet(file_name)
    
    end_time = time.time()
    print(f"Processed {final_df.height:,} elite rows in {round(end_time - start_time, 4)} seconds.")
    print("\nFirst 5 Rows of Engineered Marketing Insight:")
    
    # We select just a few columns to print so it fits nicely in the terminal
    preview_columns = ["customer_id", "traffic_source", "ctr", "roas", "clv_proxy", "customer_segment"]
    print(final_df.select(preview_columns).head(5))

if __name__ == "__main__":
    engineering_features()