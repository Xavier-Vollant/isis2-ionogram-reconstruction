# Raw CSA samples

These four PNGs are small ISIS-2 film-frame examples copied from the local
TOPIST test fixture. That fixture identifies the frames as real ionograms from
NASA's SEC dataset and keeps the original source frames alongside its
inversion references.

The filenames use the observation timestamp from each fixture directory so the
CDF-export step can derive a valid `YYYYDDDHHMMSS` observation time. The source
filenames were:

| Included file | Original fixture file |
|---|---|
| `csa_sample_1972129061031.png` | `is2_1972129_0609-007.png` |
| `csa_sample_1972176033200.png` | `is2_1972176_0330-010.png` |
| `csa_sample_1973103040746.png` | `is2_1973103_0406-009.png` |
| `csa_verified_ksh_1972322002235.png` | `B1-35-25_ISIS_B_D-758_Image0550.png` |

They are provided for local testing and visual inspection. The full CSA
archive and NASA reference corpus are not included.

The first three examples exercise image loading and structure detection; the
film-only standardizer classifies them as `review` or `not_usable`. The fourth
is a verified usable example for the complete pipeline. Its matching NASA CDF
is `data/samples/i2_av_ksh_1972322002235_v01.cdf`.
