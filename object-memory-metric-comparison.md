# Object Memory Metric Comparison

Scene: `036bce3393`  
Match threshold: `0.5 m`

| Method | Detector-coverable object memory recall | Memory redundancy ratio |
|---|---:|---:|
| SpatialRAG (ClawS) | 0.6667 (38 / 57) | 2.1579 (82 / 38) |
| DualMap | 0.4035 (23 / 57) | 1.3125 (42 / 32) |
| HOV-SG | 0.3158 (18 / 57) | 2.2857 (80 / 35) |

## Metric Meanings

**Detector-coverable object memory recall** measures how many ground-truth
objects from the detector-coverable subset are represented by the method's
object memory. A ground-truth object counts as recalled when there is at least
one predicted memory with a compatible mapped label and a 3D anchor within the
matching threshold. Higher is better. The denominator here is `57`, not all
valid ScanNet++ objects, because this metric only scores classes that the
detector/label mapping can reasonably cover.

**Memory redundancy ratio** measures how many matched memory entries are
assigned to each matched ground-truth object:

```text
memory_redundancy_ratio =
  number_of_matched_pred_memories / number_of_matched_gt_objects
```

An ideal non-duplicated object memory is close to `1.0`: one memory entry per
real object. Values above `1.0` mean the method has duplicate or fragmented
memories for the same physical object. Lower is generally cleaner, but this
metric should be read together with recall: a method can have low redundancy
simply because it misses many objects.

## Source Reports

- SpatialRAG (ClawS): `spatial-memory-evaluation/results/claws-current-scene-object-metrics.json`
- DualMap: `spatial-memory-evaluation/results/dualmap-full-recall.json`
- HOV-SG: `spatial-memory-evaluation/results/hovsg-full-recall.json`
