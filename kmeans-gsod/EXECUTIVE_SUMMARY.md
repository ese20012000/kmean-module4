# Executive Summary
## Discovering the World's Climate Zones from Weather Data Alone

**Module 4 Assignment — K-Means Python Application**

### Purpose

This application asks whether a machine learning algorithm, given nothing but a
year of daily weather readings, can rediscover the world's climate zones without
being told where any of the weather stations are. The business value of the
technique is general: it groups things that behave alike when nobody has labelled
them in advance, which is the same problem faced by customer segmentation, fault
detection and quality control.

### Dataset

NOAA's *Global Surface Summary of the Day*, obtained from the Registry of Open
Data on AWS (`s3://noaa-gsod-pds`). The dataset holds daily readings from over
12,000 weather stations worldwide. The application sampled 531 stations spread
across all 90 international reporting regions and retained 273 that had complete
enough records for 2023. The data is public and free; no AWS account was needed.

### Approach

Each station's full year of daily readings was condensed into ten descriptive
numbers — average temperature, how much the temperature swings between summer
and winter, rainfall, humidity, snow, wind, fog and thunder. K-Means then grouped
the stations by similarity across those ten measures.

The critical design decision was to **withhold each station's location**.
Latitude, longitude and elevation were never given to the algorithm. They were
kept back as an independent test: if the groups turn out to match geography
anyway, the algorithm found something genuinely present in the weather rather
than being handed the answer.

### Findings

The algorithm settled on **five climate groups**, and each is immediately
recognisable:

| Group | Stations | Character |
|---|---:|---|
| Polar / tundra | 23 | Freezing, damp, snow on a third of days |
| Subarctic continental | 81 | Freezing, with a 40 °C swing across the year |
| Arid / semi-arid | 31 | 211 mm of rain, very dry air, high altitude |
| Maritime temperate | 114 | Mild, 1,011 mm of rain, moderate seasons |
| Tropical humid | 24 | 27 °C year-round, 2,802 mm of rain, frequent thunder |

**The headline result: cluster membership explains 63.8% of the variation in
distance from the equator, and the model never saw a single coordinate.** The
stations most typical of each group confirm it — Russian Arctic outposts and an
Antarctic base in the polar group, Siberia and Fairbanks in subarctic, three
cities on the Gobi Desert margin in arid, and Colombo, Manila and
Thiruvananthapuram in tropical.

### Challenges

**Hidden missing data was the main obstacle, and it produced convincing but wrong
results twice.** NOAA does not leave gaps blank. A missing temperature is written
as 9999.9, which turned one station's annual average into several hundred
degrees. Worse, when a station reports no rainfall data at all, the file records
`0.00` and marks it only with a flag buried in a separate column. This affected a
fifth of all station-days and reported zero annual rainfall for a Scottish
Highland pass that receives over 1,000 mm. Both faults were fixed and are now
covered by automated tests so they cannot return.

**Rainfall initially overwhelmed everything else.** Because tropical totals are
many times larger than desert ones, rainfall dominated the similarity
calculation, placing a Norwegian station in the same group as the tropics simply
because both are wet. Compressing rainfall onto a logarithmic scale restored
balance.

**The obvious way to choose the number of groups turned out to be unreliable.**
The standard silhouette measure was essentially flat across every option and its
technical "best" answer differed from the runner-up by 0.0007 — noise, not
evidence. The number of groups was instead taken from the point of diminishing
returns on the inertia curve. After the data faults were corrected, three
independent measures agreed on five groups.

### Conclusion

Weather behaviour alone is enough to reconstruct the broad shape of world
geography. Two caveats matter for anyone acting on the output. Climate is a
continuum, not a set of separate boxes: 42% of stations fall into the broad
temperate group, and 4% sit so near a boundary that their assignment could
reasonably go either way — two stations at the same Montreal airport, reporting
within 0.2 °C of each other, ended up in different groups. And these results
describe 2023 only, not a long-term climate normal.
