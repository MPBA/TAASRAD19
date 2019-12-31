import argparse
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract weak labels from daily descriptions."
    )

    parser.add_argument("radar_directory", type=str)
    args = parser.parse_args()

    date_description_path = os.path.join(args.radar_directory, 'daily_weather_report.csv')
    output_path = os.path.join(args.radar_directory, 'daily_weather_report_tags.csv')

    df = pd.read_csv(date_description_path, index_col=None)
    df = df.fillna("")
    df["text"] = df.text.str.lower()

    tags = {
        "rain": r"precip\w+\b|piov\w+\b|piog\w+\b",
        "hail": r"grand",
        "storm": r"temporal",
        "downpour": r"rovesc",
        "snow": r"nev",
    }

    for tag, kw in tags.items():
        kw_r = r"\b{}\w*\b".format(kw)

        df[f"tag_{tag}"] = df.text.str.contains(kw_r)
        df[f"extract_{tag}"] = df.text.str.extract(f"({kw_r})")
        df.loc[df[f"tag_{tag}"], f"l_{tag}"] = tag

        print(tag, df[f"tag_{tag}"].sum())

    for tag, kw in tags.items():
        print(tag, ": ", kw)
        print(df[f"extract_{tag}"].dropna().unique())
        print()

    cs = [c for c in df.columns if "l_" in c]
    df[cs] = df[cs].fillna(" ")

    df["tags"] = ""
    for c in cs:
        df["tags"] += df[c] + " "

    df["tags"] = df.tags.map(lambda s: " ".join(s.split()).strip())
    df.tags.drop_duplicates()

    df.to_csv(output_path, index=False)
