import polars as pl
from faker import Faker
import random
import time
from datetime import datetime, timedelta
import numpy as np

# Initialize global Faker
fake = Faker(['en_US', 'en_GB', 'en_IN', 'fr_FR', 'es_ES', 'pt_BR', 'ja_JP'])

def generate_raw_data(num_rows=1000000):
    print(f"Generating {num_rows} rows of elite e-commerce & digital marketing data...")
    start_time = time.time()
    
    data = []
    
    # Industry-standard categories
    traffic_sources = [
        'Meta Ads', 'Google Ads', 'TikTok Ads', 
        'LinkedIn Ads', 'Email Promo', 
        'Organic Search', 'Direct']
    
    # Weighted towards mobile
    device_types = [
        'Mobile', 'Mobile', 'Mobile', 
        'Desktop', 'Tablet'] 
    
    genders = ['Male', 'Female', 
               'Non-Binary', 'Unknown']
    
    # Real time global locations simulating real messy data entries
    global_locations = [
        'Accra', 'accra', ' ACCRA ', 'GH', 
        'Lagos', 'lagos, ng', 'NIGERIA', 
        'Cape Town', 'za', 'London', 'london, UK', 
        'New York', 'NY ', 'usa', 'Tokyo', ' TOKYO']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for _ in range(num_rows):
        # --- 1. IDENTIFIERS & DEMOGRAPHICS ---
        customer_id = fake.uuid4()
        interaction_date = fake.date_between(start_date=start_date, end_date=end_date)
        age = random.choice([-1, 999, None]) if random.random() < 0.05 else random.randint(18, 65)
        gender = random.choice(genders)
        
        # Location Chaos Logic
        loc_choice = random.random()
        if loc_choice < 0.15: base_location = random.choice(global_locations)
        elif loc_choice < 0.20: base_location = random.choice(['unknown', 'null', None])
        else: base_location = fake.city() if random.random() < 0.5 else fake.country()
            
        if base_location and base_location not in ['unknown', 'null']:
            mess_factor = random.random()
            if mess_factor < 0.1: location = base_location.lower()
            elif mess_factor < 0.2: location = base_location.upper()
            elif mess_factor < 0.3: location = f" {base_location} "
            else: location = base_location
        else:
            location = base_location

        # --- 2. TRAFFIC & ENGAGEMENT ---
        traffic_source = random.choice(traffic_sources)
        device_type = random.choice(device_types)
        
        # Generating realistic advertising funnel numbers
        if traffic_source in ['Organic Search', 'Direct']:
            ad_impressions = 0
            ad_clicks = 0
            campaign_cost = 0.0
        else:
            ad_impressions = random.randint(100, 50000)
            # Realistic CTR is usually 0.5% to 5%
            ctr = random.uniform(0.005, 0.05)
            ad_clicks = int(ad_impressions * ctr)
            
            # Google Ads often uses micros, others use standard floats
            if traffic_source == 'Google Ads':
                campaign_cost = round(random.uniform(2.0, 45.0), 2) * 1000000 
            else:
                campaign_cost = round(random.uniform(1.0, 30.0), 2)

        session_duration_sec = random.randint(5, 1200) # 5 seconds to 20 minutes
        social_shares = random.randint(0, 5) if random.random() < 0.3 else 0 # 30% chance they share
        promo_sensitivity = round(random.uniform(0.0, 1.0), 2)

        # --- 3. CONVERSION & FINANCIALS ---
        # 3% to 15% conversion rate based on clicks
        converted = True if ad_clicks > 0 and random.random() < 0.08 else False
        if traffic_source in ['Organic Search', 'Direct'] and random.random() < 0.03:
            converted = True # Small chance of organic conversion
            
        if converted:
            cart_abandonment = False
            purchase_value = round(random.uniform(15.0, 850.0), 2)
            # Simulate string commas for messy data handling later
            if random.random() < 0.15: purchase_value = f"{purchase_value:,}"
            historical_purchases = random.randint(1, 24)
        else:
            # If they didn't convert, did they abandon the cart? (40% chance if they stayed > 60 secs)
            cart_abandonment = True if session_duration_sec > 60 and random.random() < 0.4 else False
            purchase_value = 0.0
            historical_purchases = random.randint(0, 5) # They might have bought in the past

        # --- 4. ATTITUDE & RETENTION ---
        # Did they leave a CSAT (Customer Satisfaction) score? (Only 20% of users do)
        csat_score = random.randint(1, 10) if random.random() < 0.2 else None
        
        # Churn Flag: 1 if they haven't bought recently and have low session time
        churn_flag = 1 if historical_purchases > 0 and not converted and session_duration_sec < 30 else 0

        data.append({
            "customer_id": customer_id,
            "interaction_date": interaction_date,
            "age": age,
            "gender": gender,
            "location": location,
            "device_type": device_type,
            "traffic_source": traffic_source,
            "ad_impressions": ad_impressions,
            "ad_clicks": ad_clicks,
            "campaign_cost": campaign_cost,
            "session_duration_sec": session_duration_sec,
            "social_shares": social_shares,
            "promo_sensitivity": promo_sensitivity,
            "cart_abandonment": cart_abandonment,
            "conversion_status": converted,
            "purchase_value": purchase_value,
            "historical_purchases": historical_purchases,
            "csat_score": csat_score,
            "churn_flag": churn_flag
        })

    # Convert to Polars and save as highly compressed Parquet
    df = pl.DataFrame(data)
    file_name = "raw_marketing_data.parquet"
    df.write_parquet(file_name)
    
    end_time = time.time()
    print(f"Raw dataset created: {num_rows} rows in {round(end_time - start_time, 2)} seconds.")
    print(f"Saved as: {file_name}")

if __name__ == "__main__":
    generate_raw_data(1000000)