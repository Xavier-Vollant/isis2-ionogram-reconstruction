# Raw CSA samples

These PNGs are small ISIS-2 film-frame examples. They are included for
local tests and visual inspection; the full CSA archive is not included.

The filenames use the observation timestamp so the CDF-export step can derive
a valid `YYYYDDDHHMMSS` observation time. The source filenames were:

| Included file | Original file |
|---|---|
| `csa_sample_1972129061031.png` | `is2_1972129_0609-007.png` |
| `csa_sample_1972176033200.png` | `is2_1972176_0330-010.png` |
| `csa_sample_1973103040746.png` | `is2_1973103_0406-009.png` |
| `csa_verified_bur_1973077231124.png` | `B1-35-31 ISIS B D-1096/Image0278.png` |
| `csa_verified_bur_1973077111224.png` | `B1-35-31 ISIS B D-1096/Image0245.png` |
| `csa_verified_bur_1973078095224.png` | `B1-35-31 ISIS B D-1096/Image0318.png` |

The first three examples exercise image loading and structure detection. The
film-only standardizer classifies them as `review` or `not_usable`. They were
copied from a local TOPIST test fixture.

The verified BUR examples are usable end-to-end. The selected default
`csa_verified_bur_1973078095224.png` is the example used by the README and
notebooks. It was recorded at Burrangong (BUR) on 1973-03-19. Its matching NASA
CDF is `data/samples/i2_av_bur_1973078095224_v01.cdf`. It was downloaded from the CSA
open data portal by `scripts/dataset/download_candidate_data.py`:

```text
https://donnees-data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/Alouette-ISIS/ISIS-2/b3_R014207773/B1-35-31%20ISIS%20B%20D-1096/Image0318.png
```

## How the default example was chosen

The selected default is gallery item #09 from the ranked comparison. It is
candidate rank 25 and the ninth highest-ranked candidate that passes the
film-only `usable` gate:

| detail | selected default |
|---|---|
| gallery item | #09 |
| candidate rank | 25 |

The CSA scan and NASA record are 2.4 seconds apart, so they are adjacent
soundings rather than the same sounding. These metrics are useful for choosing
a clear example, but they are not a claim that the model reproduces every
measured amplitude.
