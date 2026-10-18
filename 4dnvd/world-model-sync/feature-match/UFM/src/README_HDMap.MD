
Visualize VISER
```bash
pixi run python src/preprocess_cosmos_viewer.py --config src/config_geometry/monocular_shift/nuscene.yaml --splits val --port 8082
```

Please run Render to generate training/testing data
```bash
pixi run python src/preprocess_cosmos_render.py   --config src/config/double_reprojection/nuscene.yaml   --frames-only
pixi run python src/preprocess_cosmos_render.py   --config src/config_geometry/monocular_shift/nuscene.yaml   --frames-only
```
For MP4s + PNGs, replace `--frames-only` with `--save-frames`.

For offline loading:
```bash
pixi run python src/preprocess_cosmos_render.py \
  --replay-scene ./hdmap_render/scene_data/scene-0571.json \
  --frames-only \
  --output-dir ./offline_render
  ```