# Raw CSA samples

These four PNGs are small ISIS-2 film-frame examples. They are included for
local tests and visual inspection; the full CSA archive is not included.

The filenames use the observation timestamp so the CDF-export step can derive
a valid `YYYYDDDHHMMSS` observation time. The source filenames were:

| Included file | Original file |
|---|---|
| `csa_sample_1972129061031.png` | `is2_1972129_0609-007.png` |
| `csa_sample_1972176033200.png` | `is2_1972176_0330-010.png` |
| `csa_sample_1973103040746.png` | `is2_1973103_0406-009.png` |
| `csa_verified_bur_1973077231124.png` | `B1-35-31 ISIS B D-1096/Image0278.png` |

The first three examples exercise image loading and structure detection. The
film-only standardizer classifies them as `review` or `not_usable`. They were
copied from a local TOPIST test fixture.

The fourth is the verified usable example used by the README quick start, the
notebooks, and the continuous-integration smoke test. It was recorded at
Burrangong (BUR) on 1973-03-18. Its matching NASA CDF is
`data/samples/i2_av_bur_1973077231124_v01.cdf`. It was downloaded from the CSA
open data portal by `scripts/dataset/download_candidate_data.py`:

```text
https://donnees-data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/Alouette-ISIS/ISIS-2/b3_R014207773/B1-35-31%20ISIS%20B%20D-1096/Image0278.png
```

## How the verified example was chosen

Every one of the 300 matched CSA/NASA pairs in the candidate table was run
through `scripts/pipeline/run_scan.py` and scored against its paired NASA CDF
with the overlap metric used in notebook 03. 117 pairs passed the quality
gate. This example was selected for having a strong echo trace that is clearly
visible in the calibrated scan, together with near-best agreement:

| | this example | median of the 117 | best of the 117 |
|---|---|---|---|
| correlation | 0.823 | 0.770 | 0.827 |
| MAE | 0.083 | 0.095 | 0.068 |

It is an above-average case, not a typical one. Its CSA scan and NASA record
are 2.6 s apart, close to the corpus median, so they are adjacent soundings
rather than the same sounding. Amplitude correlation alone is a weak guide
here: the highest-correlation pair in the corpus has a much fainter trace,
because background agreement can dominate the statistic.
