"""Extract the parts of Bisbee et al. (2024) PA_replication.zip (Harvard Dataverse
doi:10.7910/DVN/VPN481, CC0) that the ladder needs: code, logs, README, the raw synthetic
thermometer files and the ANES files. Skips the prepped RDS files, the explanation embeddings,
the Falcon json shards and the figures. Writes the zip's SHA-256 to SOURCE.txt."""
import hashlib
import os
import zipfile

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "bisbee2024")
Z = os.path.join(D, "PA_replication.zip")
KEEP_DATA = {"therm_ANES.csv", "therm_ANES_generic.csv", "therm_ANES_RR1.csv", "therm_ANES_RR1_GPT4.csv",
             "therm_ANES_RR1_replication.csv", "therm_ANES_RR1_replication_firstperson.csv",
             "therm_ANES_RR1_replication_July.csv", "therm_temperature.csv", "anes_simp.csv",
             "anes_timeseries_cdf_stata_20220916.dta", "therm_ANES_RR1_falcon.csv"}

h = hashlib.sha256()
with open(Z, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
with zipfile.ZipFile(Z) as z:
    for n in z.namelist():
        base = os.path.basename(n)
        if n.endswith("/"):
            continue
        keep = (n.startswith("PA_replication/code/") and "/groundhog/" not in n) or base == "README_v1.txt" \
            or base == "0_master.R" or n.startswith("PA_replication/output/tables/") or base in KEEP_DATA
        if keep:
            z.extract(n, D)
            print(f"{z.getinfo(n).file_size / 1e6:9.1f} MB  {n}", flush=True)
with open(os.path.join(D, "SOURCE.txt"), "w", newline="\n") as f:
    f.write("Bisbee, Clinton, Dorff, Kenkel and Larson (2024), Political Analysis.\n"
            "Harvard Dataverse doi:10.7910/DVN/VPN481, version 1 (released 2024-04-03), CC0 1.0.\n"
            "File PA_replication.zip, datafile id 10056990, 2130570160 bytes, downloaded 2026-10-02.\n"
            f"SHA-256 {h.hexdigest()}\n")
print("sha256", h.hexdigest())
