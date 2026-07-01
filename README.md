# Super Resolution Benchmark

Reproducible benchmark comparing ESPCN, FSRCNN, and AnyUp on COCO for downstream traffic sign detection (GTSDB, TT100K).

## Models
- ESPCN
- FSRCNN
- AnyUp (arbitrary upscaling)

## Dataset
- COCO (source images)

## Structure
- configs/  � per-model and dataset YAML configs
- src/      � all pipeline code
- 
## Notebooks/ � EDA and results analysis
- scripts/  � entrypoints for train/eval/benchmark
- Artifacts/ � saved model weights, metrics
- Tests/    � unit tests


