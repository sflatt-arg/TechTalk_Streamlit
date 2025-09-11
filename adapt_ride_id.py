import pandas as pd
import ast

# File path
file_path = "tables/user_feedback.csv"

# Read CSV
df = pd.read_csv(file_path)

# Replace ride_id with incremental values in Rxxxx format
#df["ride_id"] = [f"R{i:04d}" for i in range(1, len(df) + 1)]

# Convert issues to list
#df["issue"] = df["issue"].apply(lambda x: [item.strip() for item in x.split(',')])


def to_issue_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        # try to parse "['battery', 'brakes']"
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(i).strip() for i in v if str(i).strip()]
            except Exception:
                pass
        # fallback: comma-separated
        return [p.strip() for p in s.split(",") if p.strip()]
    return [str(x).strip()]


breakpoint()
# Save back to same file
df.to_csv(file_path, index=False)