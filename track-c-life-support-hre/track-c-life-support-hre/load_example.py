import pandas as pd

df = pd.read_csv("mars500_eclss_scaled_100p_730d.csv")
storm = df[df["dust_storm_flag"] == 1]
print("days", len(df), "crew", df["crew_size"].iloc[0])
print("storm days", len(storm))
print("total O2 kg", df["o2_consumption_kg"].sum())
print("peak cabin CO2 ppm", df["cabin_co2_ppm_proxy"].max())
