# Scope of the speed-check evaluation: what was not evaluated

This file lists the evaluations of the speed check that the SP2UBI paper does not contain, with the reason for each.
Section S6 of the supplementary material summarizes it. The first two items were secondary variants of the evaluation
design that was fixed in advance (the timestamped design records of Section S2 of the supplement); they were not run,
and no reported result depends on them.

| Not evaluated | Reason and what was evaluated instead |
|---|---|
| Near-saturated urban traffic (degree of saturation 0.9) | A secondary demand variant of the design. The evaluated urban demand has a degree of saturation of 0.7, which is an assumption of the traffic model (Section S11.1 of the supplement). |
| Repeated sensitivities of the channel generator (other forms of scatterer motion, vehicle antenna pattern, IDM car following) | Secondary variants of the design. The generator applies the scatterer-motion clause of TR 37.885 as specified (per ray), the vehicle pattern is that of the evaluated vehicle antenna, and car following uses the Krauss model of the traffic simulator (Section S11.1 of the supplement). |
| Decision rates of the consistency rule under TR 37.885 | Replaced by the frequency-only separation test (Fig. 8 of the paper). The decision rates of the rule exist under our own channel model only, and the paper marks them so (Section S4 of the supplement). |
| Street-canyon channel models of TR 38.901 | TR 37.885 is the 3GPP evaluation methodology for the vehicle-to-roadside link of the paper. |
| Vehicle classes other than the passenger car | The evaluation covers the OBU antenna of a passenger car (1.6 m). Other classes need their own survey or are not covered. |
| Ray-traced case study | Not run; the evaluation uses the stochastic channel models on its primary sites. |
| Highway crossings with blockage by vehicles in the nearer lane | Not simulated. The paper states this limitation in Section VI-D and in the paragraph "Limitations and future work" of Section VIII. |
