"""
Concept Extraction Module for Spatial Concept Bottleneck Models
===============================================================
A framework for learning spatial concept masks from Broden annotations
on top of frozen DINO/SAM features.

Modules:
    config   - Argument parsing and hyperparameter management
    dataset  - Broden dataset loading and preprocessing
    model    - ConceptSegmentPredictor architecture
    loss     - ConceptSegmentationLoss with orthogonality regularization
    metrics  - IoU and evaluation utilities
    probe    - Linear probe baseline
    trainer  - Training and validation loops
"""
