# Data

The risk model and the simulated daily records use the synthetic telematics data set of

> B. So, J.-P. Boucher and E. A. Valdez, "Synthetic Dataset Generation of Driver Telematics," *Risks*, vol. 9, no. 4,
> art. 58, 2021, doi:10.3390/risks9040058.

The data set (100,000 policies, 52 variables, file `telematics_syn-032021.csv`) is published by its authors on
<http://www2.math.uconn.edu/~valdez/data.html> under their own terms; it is not part of this repository.

```
bash data/fetch_so2021.sh
```

downloads it into this folder and checks it against `SHA256SUMS`. Every result in this repository was computed from the
file with that SHA-256. Without the file, the tests that need it are skipped and the scripts of `experiments/dating/`
and `experiments/risk_model/` cannot run; everything else works without it.
