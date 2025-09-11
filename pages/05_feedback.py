import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
import tiktoken

# Charts (simple)
import altair as alt

# Supabase 2.x client
from supabase import Client
from supabase_client import supabase

# OpenAI (v1.x) and client
from openai import OpenAI
from openai_client import openai_client


# Optional: token counter (tiktoken) with fallback
def count_tokens(texts: List[str], model: str = "gpt-4o-mini") -> int:
    """
    Return an approximate token count for a list of strings.
    Tries tiktoken; falls back to rough estimate (~1 token per 4 chars).
    """
    try:
        # There's no official encoder for 4o yet; cl100k_base is a decent proxy
        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(t)) for t in texts)
    except Exception:
        return sum(max(1, len(t) // 4) for t in texts)

def chunk_comments_by_tokens(
    comments: List[str],
    step: int = 20,
    token_threshold: int = 50_000,
    model: str = "gpt-4o-mini",
) -> List[List[str]]:
    """
    Walk through comments in blocks of `step` (20 by default).
    Keep a running token total; when adding the next block would make the
    total exceed `token_threshold`, start a new chunk.

    Returns: list of chunks, each a list of comment strings.
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    running_tokens = 0

    # Iterate 20-by-20
    for i in range(0, len(comments), step):
        block = comments[i : i + step]
        block_tokens = count_tokens(block, model=model)

        if running_tokens + block_tokens > token_threshold and current:
            chunks.append(current)
            current = []
            running_tokens = 0

        current.extend(block)
        running_tokens += block_tokens

    if current:
        chunks.append(current)
    return chunks

def llm_summarize(texts: List[str], client: OpenAI, model: str = "gpt-4o-mini") -> str:
    """
    Summarize a list of comments with OpenAI, returning plain text.
    """
    if not texts:
        return "No comments provided."
    system = (
        "You are a concise analyst. Only use the provided comments. "
        "Identify key themes with quantitative estimators (approximate count of users) and propose which actions must be prioritized."
    )
    user = "Summarize the following comments:\n\n" + "\n".join(f"- {t}" for t in texts)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def llm_map_reduce(comments: List[str], client: OpenAI, model: str, step: int, token_threshold: int) -> str:
    """
    Map: summarize each chunk.
    Reduce: summarize the partial summaries into one.
    """
    if not comments:
        return "No comments to summarize with current filters."

    chunks = chunk_comments_by_tokens(
        comments, step=step, token_threshold=token_threshold, model=model
    )

    # Map
    partials = []
    for idx, chunk in enumerate(chunks, start=1):
        st.write(f"Chunk {idx}: {len(chunk)} comments")
        partials.append(llm_summarize(chunk, client, model=model))

    if len(partials) == 1:
        return partials[0]

    # Reduce
    system = (
        "You are a concise analyst. Merge these partial summaries into one. I would like you to write one or two paragraphs that synthesize the main points and highlight the most important themes. "
        "Deduplicate themes, keep counts approximate (do not hesitate to make it general, using 'many', 'a few users', 'many users', 'tenths of', ...). Second, analyze the whol picture to output 1-2 prioritized actions."
        "The whole summary should be concise and easy to read, in 150-200 words."
    )
    user = "Combine the following chunk summaries into a single coherent summary:\n\n" + \
           "\n\n".join(f"--- Chunk {i+1} ---\n{txt}" for i, txt in enumerate(partials))

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# -------------------------------
# Data loading
usr_feedback_table = supabase.table("usr_feedback").select("*").execute()
df = pd.DataFrame(usr_feedback_table.data)
# -------------------------------

# Normalize issue -> list[str]
def to_issue_list(x):
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]  # Clean existing lists
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x[1:-1]  # Remove brackets
        x = [item.strip().replace("'", "") for item in x.split(',')]
        return [item for item in x if item]  # Remove empty items
    return []  # Return empty list instead of "error"

df["issue"] = df.get("issue", "").apply(to_issue_list)

# -------------------------------
# Filters (top of page)
# -------------------------------
st.subheader("Filters")

# ride_duration: numeric range slider
min_dur = df["ride_duration"].min()
max_dur = df["ride_duration"].max()
dur_range = st.slider("Ride duration (minutes)", min_dur, max_dur, (min_dur, max_dur))

# ride_day: multiselect
days = sorted([d for d in df["ride_day"].dropna().unique().tolist() if isinstance(d, str)])
day_sel = st.multiselect("Ride day", options=days, default=days)

# satisfaction: multiselect (1..5)
sat_values = sorted([int(s) for s in df["satisfaction"].dropna().unique().tolist()])
default_sats = sat_values if sat_values else [1,2,3,4,5]
sat_sel = st.multiselect("Satisfaction (1–5)", options=[1,2,3,4,5], default=default_sats)

# issue: multiselect - FIXED TO HANDLE INDIVIDUAL ISSUES
# Extract all individual issues from all combinations
all_individual_issues = set()
for issue_list in df["issue"]:
    if isinstance(issue_list, list):
        all_individual_issues.update(issue_list)

issues = sorted(list(all_individual_issues))
issue_sel = st.multiselect("Issue", options=issues, default=issues)

# Apply filters
duration_mask = df["ride_duration"].between(dur_range[0], dur_range[1])
day_mask = df["ride_day"].astype(str).isin(day_sel)
satisfaction_mask = df["satisfaction"].astype(int).isin(sat_sel)

# FIXED: Issue filtering to check if any selected issue is in the row's issue list
if issue_sel:
    issue_mask = df["issue"].apply(
        lambda x: any(issue in x for issue in issue_sel) if isinstance(x, list) else False
    )
else:
    issue_mask = pd.Series([True] * len(df), index=df.index)

mask = duration_mask & day_mask & satisfaction_mask & issue_mask
fdf = df.loc[mask].copy()

st.caption(f"Filtered rows: {len(fdf)}")

# -------------------------------
# Three simple plots in columns
# -------------------------------
st.subheader("Quick visuals (filtered)")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Average grade**")
    avg_grade = fdf["satisfaction"].mean() if len(fdf) else 0
    st.metric("Average satisfaction", f"{avg_grade:.2f}")
    # Optional: single-bar chart for average
    avg_df = pd.DataFrame({"metric": ["avg"], "value": [avg_grade]})
    chart_avg = alt.Chart(avg_df).mark_bar().encode(x="metric", y="value")
    st.altair_chart(chart_avg, use_container_width=True)

with col2:
    st.markdown("**Histogram of grades**")
    if len(fdf):
        hist_grade = fdf["satisfaction"].value_counts().sort_index().reset_index()
        hist_grade.columns = ["satisfaction", "count"]
        chart_hist_grade = alt.Chart(hist_grade).mark_bar().encode(
            x=alt.X("satisfaction:O", title="Satisfaction"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["satisfaction","count"]
        )
        st.altair_chart(chart_hist_grade, use_container_width=True)
    else:
        st.info("No data after filters.")

with col3:
    st.markdown("**Histogram of issues**")
    if len(fdf):
        # FIXED: Count individual issues across all filtered rows
        all_issues_in_filtered = []
        for issue_list in fdf["issue"]:
            if isinstance(issue_list, list):
                all_issues_in_filtered.extend(issue_list)
        
        if all_issues_in_filtered:
            issue_counts = pd.Series(all_issues_in_filtered).value_counts().reset_index()
            issue_counts.columns = ["issue", "count"]
            chart_issue = alt.Chart(issue_counts).mark_bar().encode(
                x=alt.X("issue:O", sort="-y", title="Issue"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["issue","count"]
            )
            st.altair_chart(chart_issue, use_container_width=True)
        else:
            st.info("No issues in filtered data.")
    else:
        st.info("No data after filters.")

# -------------------------------
# Generate summary (LLM map-reduce)
# -------------------------------
st.subheader("LLM summary of comments (filtered)")

model = "gpt-4o-mini"
token_threshold = 50_000
step = 20

comments = fdf["comment"].astype(str).tolist()

if not openai_client:
    st.info("No OpenAI client created.")
else:
    client = openai_client
    if st.button("Generate summary", type="primary", disabled=len(comments) == 0):
        if not comments:
            st.warning("No comments to summarize with current filters.")
        else:
            with st.spinner("Summarizing filtered comments..."):
                summary = llm_map_reduce(
                    comments=comments,
                    client=client,
                    model=model,
                    step=int(step),
                    token_threshold=int(token_threshold),
                )
            st.text_area("Summary", summary, height=260)