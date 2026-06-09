# Chronos Fine-Tuning

An anomaly-aware fine-tuning framework for Chronos-2 that extends the default dataset and training pipeline with custom classes to preserve anomaly labels and implement a hinge-based loss. The model is trained to forecast normal windows accurately while producing higher prediction errors on anomalous windows, enabling effective anomaly detection.

## Features

- Custom Chronos2Dataset preserving anomaly labels
- Custom Trainer with hinge-based anomaly loss
- LoRA and Full Fine-Tuning support
- Time-series anomaly detection using Chronos-2
- Compatible with multivariate datasets
