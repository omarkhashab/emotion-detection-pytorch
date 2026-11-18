# docs/

```
docs/
├── LEARNING_GUIDE.md          step-by-step explanation of every stage of the project
├── screenshots/               Streamlit app screenshots (submission deliverable, linked from the main README)
│   ├── app_prediction.png         a real prediction with all six class probabilities
│   └── app_empty_input.png        empty input rejected with a warning
└── images/
    └── learning_guide/        diagrams used in LEARNING_GUIDE.md, numbered in reading order
        ├── 01_checkpoint_selection_per_epoch.png    §2.3  Why three splits?
        ├── 02_experiment_a_frozen_backbone.png      §3.1  Anatomy
        ├── 03_experiment_b_partial_finetuning.png   §3.1  Anatomy
        ├── 04_tensor_shapes_forward_pass.png        §4.1  One training step
        ├── 05_training_step_flow_1.png              §4.1  One training step
        ├── 06_training_step_flow_2.png              §4.1  One training step
        ├── 07_dropout_train_vs_eval.png             §4.2  model.train() vs model.eval()
        ├── 08_with_vs_without_gradients.png         §4.2  model.train() vs model.eval()
        ├── 09_train_then_eval_each_epoch.png        §4.2  model.train() vs model.eval()
        ├── 10_loss_curves_healthy.png               §4.3  Model selection
        ├── 11_loss_curves_overfitting.png           §4.3  Model selection
        ├── 12_inference_flow_1.png                  §6    Inference
        ├── 13_inference_flow_2.png                  §6    Inference
        ├── 14_saved_model_contents.png              §6    Inference
        ├── 15_model_caching.png                     §6    Inference
        ├── 16_software_architecture.png             §7    Streamlit app
        └── 17_technology_stack.png                  §9    Going further
```

The diagrams in `images/learning_guide/` are **conceptual illustrations**. Any numbers in them (for example the
epoch Macro F1 values in `01_…` or the 97% prediction in `13_…`) are examples, not measured results. The real
results are in [`../results/`](../results/README.md) and in section 6 of the [main README](../README.md).

To add a new image, put it in the matching subfolder with a numbered, descriptive name and link it with a
relative path, for example `![What it shows](images/learning_guide/18_new_diagram.png)`.
