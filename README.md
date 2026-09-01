# Global Climate Clustering with K-Means

Module 4 Assignment. A Python application that clusters weather stations
worldwide into climate types using K-Means, built on the NOAA Global Surface
Summary of the Day dataset from the Registry of Open Data on AWS.

## The question

Given a year of daily weather readings from hundreds of stations around the
world, and no information about where any of them are, can K-Means recover the
world's climate zones from the weather alone?

Latitude, longitude and elevation are read from the data but deliberately
excluded from the feature set. They are held back and used afterwards to test the
result. If clusters built purely from temperature, rainfall and humidity line up
with latitude anyway, the model found real geography rather than being told it.

## Result

243 to 273 stations across 90 WMO regions, reduced to ten climate features and
clustered into five groups:

| Cluster | Label | Stations | Mean temp | Seasonality | Rainfall | Mean \|latitude\| |
|---|---|---:|---:|---:|---:|---:|
| 0 | Polar / tundra | 23 | -1.0 °C | 20.4 °C | 621 mm | 60.5° |
| 1 | Subarctic continental | 81 | -0.5 °C | 39.7 °C | 570 mm | 56.9° |
| 2 | Arid / semi-arid | 31 | 12.4 °C | 26.9 °C | 211 mm | 35.6° |
| 4 | Maritime temperate | 114 | 16.5 °C | 15.9 °C | 1,011 mm | 36.2° |
| 3 | Tropical humid | 24 | 27.1 °C | 3.6 °C | 2,802 mm | 11.4° |

**Cluster membership explains 63.8% of the variance in absolute latitude and
20.7% in elevation, neither of which the model was shown.**

The stations nearest each centroid are recognisable: Russian Arctic outposts and
an Antarctic base in tundra; Siberia and Fairbanks in subarctic continental;
Yinchuan, Zhangye and Jiuquan on the Gobi margin at 1,408 m in arid;
Thiruvananthapuram, Manila and Colombo in tropical humid.

## Running it

**Python 3.9 or newer is required; 3.12+ recommended.** Check with
`python --version` before anything else.

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows
source .venv/bin/activate               # macOS / Linux

python -m pip install --upgrade pip     # do not skip this
pip install -r requirements.txt

python -m src.main                      # full run: downloads, clusters, plots
```

Upgrading pip is not optional housekeeping. An old pip silently discards every
package version that needs a newer interpreter than yours, then reports
`ERROR: No matching distribution found for boto3`, which points at the package
when the real problem is your Python version. If you see that, run
`python --version` first.

`requirements.txt` specifies minimum versions so it resolves across
interpreters. `requirements-lock.txt` holds the exact versions this was tested
against and needs Python 3.11+.

No AWS account, credentials or charges are needed to run the application itself.
The NOAA bucket is public and requests are unsigned.

| Option | Effect |
|---|---|
| `--year 2022` | a different observation year |
| `--stations 300` | cap the sample size |
| `--per-block 3` | stations per WMO region, controls global spread |
| `--k 6` | override automatic selection of k |
| `--offline` | reuse the local cache, no network calls |
| `--verbose` | debug logging |

First run downloads about 530 files (~25 MB) and takes two to three minutes.
Everything is cached under `data/`, so later runs are seconds.

### In PyCharm

Open this folder as a project. PyCharm detects `.venv` automatically. For the
main application, add a run configuration with **module name** `src.main` (not
script path) and the working directory set to the project root. To run the tests,
right-click the `tests` folder and choose *Run 'Unittests in tests'*.

## Tests

```bash
python -m unittest discover -s tests -t .
```

125 tests, no network access, about 35 seconds. One further test hits the live
bucket and is skipped unless you ask for it:

```bash
set RUN_NETWORK_TESTS=1 && python -m unittest tests.test_data_loader
```

The tests are not decoration. Three of them exist because they caught real bugs
that had already corrupted the results:

- `test_sentinels_do_not_poison_the_mean` — GSOD writes 9999.9 for a missing
  temperature. Untreated, a station with ten missing days averaged several
  hundred degrees.
- `test_unreported_precipitation_is_missing_not_zero` — GSOD writes `0.00` with
  flag `I` when a station reported no rainfall data at all. This affects a fifth
  of all station-days and reported 0 mm of annual rain for a Scottish Highland
  pass that receives well over 1,000 mm.
- `test_seasonal_desert_is_arid_not_continental` — the label rules tested
  seasonality before dryness, so the Iranian plateau came out as "temperate
  continental" on 211 mm of rain.

## Layout

```
src/config.py         all tunable values and the documented data quirks
src/data_loader.py    S3 listing, sampling, download with caching, cleaning
src/features.py       daily observations -> one climate fingerprint per station
src/clustering.py     scaling, k selection, fitting, validation
src/visualise.py      the four figures
src/main.py           pipeline and CLI

tests/fixtures.py         synthetic GSOD data, including the sentinel quirks
tests/test_data_loader.py acquisition and cleaning
tests/test_features.py    feature engineering and quality gates
tests/test_clustering.py  scaling, k selection, fitting, validation

outputs/              figures, labelled station table, diagnostics, run log
data/                 cached station CSVs (gitignored)
```

## Method

**Data.** `s3://noaa-gsod-pds/2023/<station>.csv`, one file per station per year.
The bucket holds 12,311 files for 2023 and has no station index, so stations
cannot be filtered by country before download. The first two digits of a GSOD
station ID are the WMO block number, which is allocated geographically, so the
application samples the largest few files from every block. Largest as a proxy
for most complete, which is deterministic and reproducible.

**Features.** Each station's year collapses to ten numbers: mean temperature,
seasonality (warmest month minus coldest), diurnal range, dew point depression,
log annual rainfall, wet-day fraction, snow-day fraction, mean wind, fog-day
fraction and thunder-day fraction. Stations with fewer than 300 days of
temperature, fewer than 200 days of rainfall, or fewer than 10 usable months are
rejected rather than characterised from thin evidence. That rejection is why 531
downloads became 273 stations.

**Rainfall is clustered on a log scale.** The raw totals span roughly 20 mm to
3,500 mm, so a monsoon station sits several standard deviations out even after
standardization and drags centroids toward itself. Standardizing equalises
variance but does nothing about skew. On the untransformed feature the clustering
was driven almost entirely by rainfall, which put a Norwegian station at 3.9 °C
in the same cluster as the genuine tropics because both are wet.

**Choosing k.** Features are standardized first, since K-Means minimises
Euclidean distance and the raw units are incomparable. k comes from the elbow of
the inertia curve, located geometrically as the point furthest below the chord
joining the ends of the curve, rather than by eye.

Silhouette was the obvious choice and was rejected on evidence. In an earlier run
it was flat across every k, spanning 0.209 to 0.235, and its maximum fell at
k=10, beating k=6 by 0.0007. That is noise, and a flat silhouette curve is the
expected result for climate, which is a continuum rather than a set of separated
blobs. Silhouette and Davies-Bouldin also tend to keep improving as k grows, so
either alone drifts to whichever end of the search range you stopped at. All
three metrics are still computed and printed next to the elbow; after the data
cleaning was corrected, all three independently agreed on k=5.

**Validation.** Explained variance (eta squared) of absolute latitude and
elevation by cluster membership, both withheld from the model.

## Limitations

- **The sample is skewed toward well-instrumented sites.** Taking the largest
  files favours international airports over rural stations.
- **WMO blocks are not proportional to land area or population.** Russia spans
  roughly twenty blocks and is over-represented; small countries share one.
- **The temperate cluster is a catch-all.** 114 of 273 stations, 42%, land in
  "maritime temperate", spanning 2.2 °C to 31.2 °C of seasonality. The world's
  temperate middle genuinely is a continuum and K-Means has to put it somewhere.
- **Boundaries are soft.** 4.0% of stations sit almost as close to a neighbouring
  centroid as their own. The clearest illustration is in the data itself: two
  separate stations at Montreal airport, Trudeau and Dorval, report within 0.2 °C
  and 0.1 °C of each other and were assigned to *different* clusters. K-Means
  draws a hard line through continuous space, and near that line the assignment
  is arbitrary.
- **One year only.** 2023 is a single sample of each station's climate, not a
  climate normal. A 30-year mean would be the meteorologically correct input.
- **The climate labels are a readability aid, not a model output.** They come
  from transparent threshold rules in `features.describe_cluster`, loosely
  following the reasoning behind the Köppen classification. The clusters are the
  result; the names just save the reader from "cluster 3".
- **Rainfall totals are conservative.** GSOD's `PRCP_ATTRIBUTES` flags record
  whether a daily total covers 6, 12 or 24 hours. This application treats
  reported values uniformly and only excludes flag `I`. Stations reporting
  6-hourly totals are therefore slightly understated.

## Next step

The same feature table feeds the Module 5 dimensionality reduction assignment:
ten correlated climate features, PCA to compress them, then clustering in the
reduced space for comparison against these results.

## Citations

NOAA National Centers for Environmental Information. (2024). *Global Surface
Summary of the Day (GSOD)*. Registry of Open Data on AWS.
https://registry.opendata.aws/noaa-gsod/ (accessed 31 August 2026).
Data accessed at `s3://noaa-gsod-pds/2023/`.

NOAA data disseminated through the NOAA Open Data Dissemination programme are
open to the public with no restrictions on use. NOAA requests attribution and
does not endorse or affiliate with this work. The data here is unmodified NOAA
data subjected to the cleaning and aggregation described above.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos,
A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011).
Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*,
12, 2825–2830.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and
validation of cluster analysis. *Journal of Computational and Applied
Mathematics*, 20, 53–65.

Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a "kneedle"
in a haystack: Detecting knee points in system behavior. *31st International
Conference on Distributed Computing Systems Workshops*, 166–171.
