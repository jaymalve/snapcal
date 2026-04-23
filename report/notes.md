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
- After the remaining experiments are completed, update these sections in `report/main.tex`:
  - `SAM Segmentation Ablation`
  - `Cross-Architecture Comparison`
  - `Calorie Estimation Accuracy`
  - the abstract and conclusion, if new results materially change the headline claims

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
serving size" this para
2. add frontend UI screenshot
3. Update confusion matrix pairs
4. Verify if its according to the requirements
5. How to remove the water mark
6. Padding in pipeline diagram header.
7. Portion size control in the frontend.
8. Fix title of the report
