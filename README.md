# torchcal: post-hoc calibration with pytorch

This package provides PyTorch implementations (including GPU support) of post-hoc calibration techniques commonly employed for deep neural networks.

Methods include Temperature, Vector and Matrix Scaling from the paper ["On calibration of modern neural networks"](https://arxiv.org/abs/1706.04599), and BCTS and NBVS from ["Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation"](https://arxiv.org/abs/1901.06852).

## install

```bash
pip install git+https://github.com/rishabh-ranjan/torchcal
```

## use

```python
cal = torchcal.calibrator("temp_scaler", device=device)

# yhat = predicted logits, y = true class labels
cal.fit(yhat_val, y_val)
print("fitted temperature = ", cal.temp.item())

yhat_test = cal(yhat_test)
```

```python
cal = torchcal.calibrator("vector_scaler", num_classes, device=device)

# yhat = predicted logits, y = true class labels
cal.fit(yhat_val, y_val)

yhat_test = cal(yhat_test)
```

Supported calibrators (and number of calibration parameters) are:
```python
[
	"temp_scaler",			# 1
	"no_bias_vector_scaler",	# num_classes
	"bias_corrected_temp_scaler",	# num_classes + 1
	"vector_scaler",		# num_classes * 2
	"no_bias_matrix_scaler",	# num_classes ** 2
	"matrix_scaler",		# num_classes ** 2 + num_classes
]
```

## cite

If you use this package, please consider citing the original papers and the [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) library:

```bibtex
@inproceedings{guo2017calibration,
  title={On calibration of modern neural networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q},
  booktitle={International conference on machine learning},
  pages={1321--1330},
  year={2017},
  organization={PMLR}
}
```

```bibtex
@inproceedings{alexandari2020maximum,
  title={Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation},
  author={Alexandari, Amr and Kundaje, Anshul and Shrikumar, Avanti},
  booktitle={International Conference on Machine Learning},
  pages={222--232},
  year={2020},
  organization={PMLR}
}
```

```bibtex
@misc{feinman2021pytorch,
  author = {Feinman, Reuben},
  title = {Pytorch-minimize: a library for numerical optimization with autograd},
  publisher = {GitHub},
  year = {2021},
  url = {https://github.com/rfeinman/pytorch-minimize},
}
```

