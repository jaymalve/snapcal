# Report Notes

- Draft placeholders are acceptable for:
  - full SAM ablation results across raw vs. segmented inputs
  - final ViT vs. ResNet vs. EfficientNet comparison
  - reliable calorie MAE
- Before submitting the final PDF, all placeholder tables, figures, and narrative claims for those three items must be replaced with real experimental results.
- Before submitting the final PDF, replace the placeholder pipeline diagram with a real architecture diagram.
- Before submitting the final PDF, replace the placeholder frontend figure with a real screenshot of the completed UI.
- The report should present the backend as fully implemented in Django.
- The report should present the portion-size control as completed in past tense because the final submission will include that frontend integration.
- The preset portion-selection feature does not require rerunning model accuracy metrics; only the latency check is worth refreshing if the report needs current runtime numbers.
- After the remaining experiments are completed, update these sections in `report/main.tex`:
  - `SAM Segmentation Ablation`
  - `Cross-Architecture Comparison`
  - `Calorie Estimation Accuracy`
  - the abstract and conclusion, if new results materially change the headline claims

## Latency Check

1. Start the backend with the production bundle and wait for the service to finish loading.
2. Run the CLI smoke test once for a cold-start style measurement:

```bash
source .venv/bin/activate
python ml/scripts/benchmark_latency.py \
  --bundle-dir artifacts/models/production_bundle \
  --image /absolute/path/to/test-image.jpg \
  --portion-unit serving
```

3. If the report needs warm-start latency, hit the running API several times and use the later requests:

```bash
curl -s -o /dev/null -w "request_1 total=%{time_total}s\n" -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "portion_unit=serving"

curl -s -o /dev/null -w "request_2 total=%{time_total}s\n" -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "portion_unit=serving"

curl -s -o /dev/null -w "request_3 total=%{time_total}s\n" -X POST http://127.0.0.1:8000/api/v1/predict \
  -F "image=@/absolute/path/to/test-image.jpg" \
  -F "portion_unit=serving"
```

4. If you want a quick sanity check that the new portion inputs do not meaningfully change runtime, repeat the same curl test with `portion_unit=oz` and `portion_value=8`, or `portion_unit=fl_oz` and `portion_value=8`.

## Next Best Steps

1. Fill the remaining experiment placeholders after the final runs.
2. Add the frontend portion-size UI change.
3. Add the pipeline diagram to the report.
4. Add a frontend screenshot to the report.
5. Compile the report and do a final wording pass on the abstract and conclusion.
6. Also check about what is mentioned about portion control frontend hook.


TODO:
1. remove "Another useful insight is that the project evolved slightly from the
original proposal while still preserving its core goal. The proposal
described a FastAPI backend, but the final production backend was
implemented in Django. That change did not weaken the system;
in practice, the Django API cleanly exposed health and prediction
endpoints and integrated well with the exported bundle format. The
project also finalized portion-size support as part of the application
workflow so that nutrition output could be scaled beyond the default
serving size" this para - done
2. add frontend UI screenshot
3. Update confusion matrix pairs
4. Verify if its according to the requirements
5. How to remove the water mark - done
6. Padding in pipeline diagram header. - done
7. Portion size control in the frontend. - done
8. Fix title of the report - done
9. Refresh latency numbers if the final report needs updated runtime measurements.
