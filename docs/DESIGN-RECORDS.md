# Timestamped design records of the speed-check evaluation

This file lists the design records of studies 1 and 2 of the paper's speed-check evaluation (Section VI-D) and their
OpenTimestamps proofs. The proofs themselves ship in the folder `stamps/` of the paper's Supplementary Material archive
(with `stamps/SHA256SUMS`); they are not part of this repository.

The rules, thresholds and design constants of studies 1 and 2 were fixed in design records before the data they
govern existed. Each record is a bundle of SHA-256 sums over the design text, the protocol constants and every tracked
source file at the tagged commit; the SHA-256 of each bundle was timestamped with OpenTimestamps, whose proof links it
to a Bitcoin block and shows that the bundle existed no later than that block. Each proof can be checked with
`ots verify -d <SHA-256> <proof>`. The site seeds of study 2 are derived from its stamped freeze bundle and the block
that attests it, so they were not known when the design was frozen.

```
Bundle (proof in stamps/)                                        Recorded (UTC)        Commit    Block   Block time (UTC)
Study 1, rules and development pilot, before any evaluation data
  (study1-rules-and-pilot.ots)                                   2026-10-21T19:44:05Z  abcf0be8  972525  2026-10-21T20:16:54Z
Study 1, freeze, before the evaluation run (study1-freeze.ots)   2026-10-24T14:09:00Z  2a718593  972929  2026-10-24T14:20:04Z
Study 2, development step, before the development data were
  analyzed (study2-step0.ots)                                    2026-10-30T16:58:06Z  f0bcab82  973857  2026-10-30T18:09:18Z
Study 2, freeze, before the run on the held-out sites
  (study2-freeze.ots)                                            2026-11-01T19:53:39Z  a07d8dc1  974170  2026-11-01T20:19:26Z
```

Recorded: time of the record (for study 2, the time the bundle was submitted to the calendars). Commit: first eight
hexadecimal digits of the tagged commit.
