import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
url = "https://files.data.gouv.fr/geo-dvf/latest/csv/"

page = requests.get(url).text
soup = BeautifulSoup(page, "html.parser")
all_years = [
    url + node.get("href")
    for node in soup.find_all("a")
    if ".." not in node.get("href")
]

communes_dict = {
    # "Roscanvel": {"depart": 29, "ninsee": 29238},
    # "Camaret": {"depart": 29, "ninsee": 29022},
    # "Lanveoc": {"depart": 29, "ninsee": 29120},
    # "Crozon": {"depart": 29, "ninsee": 29042},
    # "Telgruc": {"depart": 29, "ninsee": 29280},
    # "Argol": {"depart": 29, "ninsee": 29001},
    # "Landevennec": {"depart": 29, "ninsee": 29104},
    "Brest": {"depart": 29, "ninsee": 29019},
    "Lorient": {"depart": 56, "ninsee": 56121},
}
all_csv = []
dfs = []
for y in all_years:
    for key, val in communes_dict.items():
        df = pd.read_csv(
            "/".join(
                [
                    y,
                    "communes",
                    str(val["depart"]),
                    str(val["ninsee"]) + ".csv",
                ]
            )
        )

        df["date_mutation"] = pd.to_datetime(df["date_mutation"])

        filter = (df["type_local"] == "Appartement") & (df["valeur_fonciere"] < 4e5)
        filter &= df["surface_reelle_bati"] > 60
        filter &= df["surface_reelle_bati"] < 80
        df = df[filter]
        groups = df.select_dtypes(exclude=np.number).columns.tolist()
        dfs.append(df)


df = pd.concat(dfs)
df["prixm2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
df["Month"] = df.apply(lambda x: x["date_mutation"].month, axis=1)
df["Year"] = df.apply(lambda x: x["date_mutation"].year, axis=1)


def high_pass_filter(data, cutoff_freq, fs):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(5, normal_cutoff, btype="high", analog=False)
    y = filtfilt(b, a, data)
    return y



df.set_index("date_mutation", inplace=True)
df_resampled = (
    df.groupby("nom_commune")[["valeur_fonciere", "prixm2"]].resample("D").mean()
)
df_resampled = df_resampled.groupby("nom_commune").transform(
    lambda x: x.interpolate(method="linear")
)

# Apply the high-pass filter
df_resampled["prixm2_smooth"] = df_resampled.groupby("nom_commune")["prixm2"].transform(
    lambda x: x
    - high_pass_filter(x, cutoff_freq=0.025, fs=1)
)


ys = ["prixm2_smooth"]  # , "valeur_fonciere"]
for y in ys:
    sns.lineplot(
        data=df_resampled,
        x="date_mutation",
        hue="nom_commune",
        style="nom_commune",
        y=y,
    )
    plt.tight_layout()
    plt.savefig("assets/prixm2.svg")
    plt.show()
