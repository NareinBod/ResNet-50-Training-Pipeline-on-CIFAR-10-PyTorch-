# ResNet-50 Training Pipeline on CIFAR-10 (PyTorch)

A complete, high-performance PyTorch implementation of the ResNet-50 architecture for multi-class image classification on the CIFAR-10 dataset.

### Key Features:

* **Architecture:** Implements the full ResNet-50 Bottleneck Block structure.

* **Data Handling:** Includes data loading, aggressive data augmentation (`RandomCrop`, `RandomFlip`), and robust device abstraction (`DeviceDataloader`) for seamless CPU/GPU utilization.

* **Optimization:** Uses the Adam optimizer with Weight Decay (L2 regularization) and a **Cosine Annealing Learning Rate Scheduler** for superior convergence.

* **Performance:** Achieved a competitive validation accuracy of approximately **86.8%** on the CIFAR-10 validation set.

* **Training Loop:** Features professional, reusable `train` and `test` functions for clean metric tracking across epochs.
