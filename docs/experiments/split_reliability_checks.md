# Split Reliability Checks

## Purpose

This step checks whether the current sliding-window split may produce overly optimistic results due to adjacent overlapping windows appearing near split boundaries.

## Checks

1. Gap split for 3 hp → 2 hp
2. Target supervised 2 hp upper bound

## Notes

- The gap split discards a small middle region between train and validation/test partitions to reduce the influence of adjacent overlapping windows across splits.
- The target supervised upper bound is only a diagnostic experiment. It is not a fair domain adaptation baseline because it uses labeled target-domain training data.

