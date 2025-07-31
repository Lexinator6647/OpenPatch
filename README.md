# OpenPatch

A minimal, interpretable object detection system built from scratch in PyTorch. This project implements a custom two-stage architecture using dense sliding windows and feature maps, offering an alternative to popular models like YOLO, RetinaNet, or Faster R-CNN.

Unlike anchor-based models, this detector uses a grid-style sliding approach on top of convolutional feature maps, allowing for flexible window coverage with overlap and smoother detection near object boundaries. Ideal for those seeking a lightweight, transparent alternative to large detection frameworks.

This code base was trained, developed and tested off the COCO 128 training set with coco format labels in JSON. The *training.py* and *inference.py* scripts under *src/* are compatible with COCO JSON annotations such that the bounding boxes are in the format <br>
```[x_min, y_min, width, height]```

---

## Highlights

* **Sliding Window Detection**: No anchors or predefined bounding box priors. Instead, dense windows are unfolded across feature maps.
* **Two-Stage Modular Design**:

  * **Backbone CNN** to generate spatial feature maps.
  * **Window Network** to classify and regress bounding boxes per patch.
* **Built-in Loss Masking**: Applies classification loss only when an object is present.
* **Training + Inference Support**: Includes clean separation between training and inference with `torch.no_grad()` context.
* **Post-processing with NMS**: Supports custom Non-Maximum Suppression over sliding window results.
* **Compatible with PyTorch Lightning** for distributed training.

---

## Motivation

YOLO and similar detection models are powerful but increasingly complex and restrictive under certain licenses. This project was born from the need for:

* A **clean, license-friendly base** for custom detection work.
* The ability to **experiment with overlapping grids** and different window strategies.
* A fully **customizable architecture** suitable for research, prototyping, or integration in MLOps pipelines.

---

## Example Use Case

```
bash python download_coco_annotations.py
bash python import_coco_by_class.py --stage train
bash python inference.py --model my-model.pth --images images/example.png
```

---

## Current Limitations

* **Manual window size and overlap tuning**: Stil needs enhancements to automate this, estimating average window size of labelled objects.
* **Not tested at large scale yet**: Functional for small datasets and toy examples.
* **No export pipeline (yet)**: Not currently packaged for ONNX or TorchScript.

---

## Roadmap / Future Work

* [ ] Support additional annotation formats aside from COCO (Pascal VOC, YOLO)
* [ ] Add more metrics for training script

---

## License

MIT License. No forced open-source disclosure for commercial use unlike YOLOv8+.

---

## Credits

Built with PyTorch. Inspired by the architecture and sliding nature of convolution itself. Initial concept derived from rethinking object detection as a multi-window classification problem. **Author**: Alexandra Ciupahina

---

## Contact / Collaboration

This repo is actively maintained. Contributions or collaborations are welcome — especially in optimizing NMS, improving training stability, or testing edge cases. Reach out via issues or pull requests!

