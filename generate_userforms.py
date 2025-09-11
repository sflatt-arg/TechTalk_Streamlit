import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI


def get_api_key():
    return st.secrets["openai"]["api_key"]
# Load your API key from st.secrets (or environment variable)
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
# -------------------------------
# 1) Synthetic PII-safe dataset
# -------------------------------
def make_synth_feedback(n=10, seed=42):
    rng = np.random.default_rng(seed)
    user_ids = [f"U{rng.integers(1000, 9999)}" for _ in range(n)]
    ride_ids = [f"R{rng.integers(10_000, 99_999)}" for _ in range(n)]
    bike_ids = [f"B{rng.integers(1, 300)}" for _ in range(n)]
    day_of_week_pool = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    issues_pool = ["brakes", "chain", "light", "saddle", "battery", "cargo", "reservation problem", "unavailable bike", "none"]
    bike_types = ["E-Bike", "CargoBike", "CityBike"]

    ride_duration = rng.choice([30, 45, 60, 90, 120,180,240,300], size=n, p=[0.25,0.20,0.20,0.15,0.10,0.05,0.03,0.02])
    ride_day = rng.choice(day_of_week_pool, size=n, p=[0.15, 0.15, 0.15, 0.15, 0.20, 0.10, 0.10])
    ride_bike_type = rng.choice(bike_types, size=n, p=[0.3, 0.2, 0.5])

    def issue_from_day_and_type(ride_day,bike_type):
        # battery isues only for E-Bike and CargoBike, Cargo issu only for CargoBike.
        # more reservation problems and unavailable bike issues on weekends
        # more light issues on weekdays (commuting)
        #there can be no issue at all, or only one issue and one "none"
        if  bike_type == "E-Bike":
            if ride_day in ["Saturday", "Sunday"]:
                return rng.choice(issues_pool, size=2, p=[0.06, 0.06, 0.08, 0.07, 0.13, 0.0, 0.18, 0.18, 0.24])
            else:
                return rng.choice(issues_pool, size=2, p=[0.1, 0.07, 0.18, 0.08, 0.15, 0.0, 0.07, 0.1, 0.25])
        elif bike_type == "CargoBike":
            if ride_day in ["Saturday", "Sunday"]:
                return rng.choice(issues_pool, size=2, p=[0.05, 0.05, 0.07, 0.06, 0.12, 0.12, 0.18, 0.18, 0.17])
            else:
                return rng.choice(issues_pool, size=2, p=[0.1, 0.06, 0.15, 0.07, 0.12, 0.1, 0.06, 0.08, 0.26])
        elif bike_type == "CityBike":
            if ride_day in ["Saturday", "Sunday"]:
                return rng.choice(issues_pool, size=2, p=[0.08, 0.07, 0.1, 0.08, 0.0, 0.0, 0.15, 0.15, 0.37])
            else:
                return rng.choice(issues_pool, size=2, p=[0.12, 0.1, 0.2, 0.1, 0.0, 0.0, 0.08, 0.1, 0.3])
        else:
            print("Error in bike type")
            return ["None","None"]      

    def satisfaction_from_issue(issue_list):
        if issue_list.count("none") == 2:
            return rng.choice([4, 5], p=[0.2, 0.8])
        elif issue_list.count("none") == 1:
            return rng.choice([2,3,4,5], p=[0.1,0.3,0.4, 0.2])
        else:
            return rng.choice([1, 2, 3], p=[0.25, 0.4, 0.35])
        
    def build_issue_str(issue_list):
        if issue_list[0] != issue_list[1]:
            return ", ".join([iss for iss in issue_list if iss != "none"]) if any(iss != "none" for iss in issue_list) else "none"
        else:
            return issue_list[0]
        
    ride_issue = [issue_from_day_and_type(ride_day[i], ride_bike_type[i]) for i in range(n)] 
    satisfaction_ride = [satisfaction_from_issue(list(ride_issue[i])) for i in range(n)]

    system_prompt = """
    You are a customer feedback generator for a bike rental service based on satisfaction ratings, reported issues. 
    Your comments should be concise, relevant, and reflect the user's experience in 3-4 short sentences
    Do not mention the specific satisfaction rating, but reflect it in the tone of the comment.
    If no issues were reported and satisfaction is high (4 or 5), generate a positive comment.
    If issues were reported and satisfaction is low (1 or 2), generate a negative comment"""

    user_context= """
    The context for the ride : \n
    Day of the week : {ride_day} \n
    Ride duration (in minutes) : {ride_duration} \n
    Ride issues reported : {ride_issue_str}\n
    Satisfaction rating (1-5) : {satisfaction_ride}.
    """
    def mk_comment(ride_day, ride_duration, ride_issue_str, satisfaction_ride):
        # Ask a question
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # choose a model, e.g. gpt-4o, gpt-4o-mini, gpt-3.5-turbo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context.format(ride_day=ride_day, ride_duration=ride_duration, ride_issue_str=ride_issue_str, satisfaction_ride=satisfaction_ride)}
            ],
        )
        # Get the answer
        answer = response.choices[0].message.content
        return answer


    user_feedback_list = []
    for i in range(n):
        if i%20==0:
            print(f"Generating feedback {i}/{n}")

        issue_str = build_issue_str(ride_issue[i])
        user_feedback_list.append({
            "ride_id": ride_ids[i],
            "user_id": user_ids[i],
            "bike_id": bike_ids[i],
            "ride_duration": ride_duration[i],
            "ride_day": ride_day[i],
            "satisfaction": satisfaction_ride[i],
            "issue": issue_str,
            "comment": mk_comment(ride_day[i],ride_duration[i],issue_str,satisfaction_ride[i]), 
        })
    
    df = pd.DataFrame(user_feedback_list)
    return df

df = make_synth_feedback(n=500, seed=42)
df.to_csv("tables/user_feedback.csv", index=False)