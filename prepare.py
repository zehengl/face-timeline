# %%
from pathlib import Path
from os import getenv

import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

selfies = Path(getenv("selfies"))
assert selfies.exists()

output = Path("output")
output.mkdir(exist_ok=True)

# %%
files = list(selfies.glob("*.jpg"))
df = pd.DataFrame({"file": files})
df["date"] = df["file"].apply(lambda x: pd.to_datetime(x.stem))
df.to_pickle(output / "df.pkl")

# %%
plt.cla()
ax = sns.countplot(df, x=df["date"].dt.year)
ax.set(xlabel="Year")
ax.get_figure().savefig("output/count-per-year.png", dpi=300, bbox_inches="tight")

# %%
plt.cla()
ax = sns.countplot(df, x=df["date"].dt.month)
ax.set(xlabel="Month")
ax.get_figure().savefig("output/count-per-month.png", dpi=300, bbox_inches="tight")

# %%
plt.cla()
ax = sns.countplot(
    df,
    x=df["date"].dt.day_name(),
    order=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)
ax.set(xlabel="Weekday")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.get_figure().savefig("output/count-per-weekday.png", dpi=300, bbox_inches="tight")
