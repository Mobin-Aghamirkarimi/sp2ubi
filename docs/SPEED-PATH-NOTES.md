# Notes on the speed path

Notes on the speed-verification path of SP2UBI that complement Sections V-C to V-E of the paper and Section S2 of its
supplementary material (message encodings).

### Scope

The scope LTE-V2X (Release 14 PC5) is named explicitly because the U.S. rules designate the upper 30 MHz of the 5.9 GHz
band for ITS using C-V2X without fixing a 3GPP release (Federal Communications Commission, Second Report and Order FCC
24-123, ET Docket No. 19-138, released 21 Nov. 2024).

### Layer-2 identifier

The PC5 source Layer-2 identifier appears in the MAC header of every sidelink data transmission (SL-SCH MAC PDU) (3GPP
TS 36.321 V14.10.0, clauses 5.14.1.1 and 6.2.4). Drawing it at random per burst and using it for no other transmission
is stricter than the rule that it be changed over time and randomized (3GPP TS 23.285 V14.9.0, clause 4.5.1); selecting
new sidelink resources at each burst boundary keeps a semi-persistent reservation (3GPP TS 36.321 V14.10.0, clauses
5.14.1.1 and 6.2.4) from linking beyond the burst.

### Re-randomization at release

Without the re-randomization of every component at release, $\mathrm{Enc}(\tilde b)$ would be a combination of broadcast
ciphertexts that anyone knowing the RSU's coefficients could recompute.

### Expected number of records and the conflict case

A policy is expected to send one record per insured day in the default configuration and one per epoch in the monthly
configuration; a record carries no day index. In the conflict case the dispute procedure ends without an opening
(Section V-E of the paper).

### Burst length and beacon gate

A shorter burst would shorten the stretch over which crossings are linked but would lose more crossings. Where a
deployment uses the beacon gate of Section V-D4 of the paper, the OBU starts the beacons at the start of a trip after a
random delay of up to one burst, so the stream of a trip begins up to one burst after the trip itself; the end of a trip
stays visible. The evaluation does not model this gate, which can lose a crossing shortly after a trip starts.
