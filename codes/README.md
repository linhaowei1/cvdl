# README

A image recognition task.

Imagenette : see https://github.com/fastai/imagenette

model: wide resnet50-2

For training：

```python
python runner.py --cuda='cuda:2' --model='wide_resnet50_2' --save_path='./wide_resnet50_2_from_scratch_lr=4.pth' --log_dir='log_wide_resnet50_2_from_scratch_lr=4' --epochs=500 --lr=1e-4 --batch-size=32 --mode='train' --seed=1111
```

For testing:

```python
python runner.py --cuda='cuda:4' --model='wide_resnet50_2' --model_params_path='./wide_resnet50_2_from_scratch_lr=4.pth' --log_dir='log_wide_resnet50_2_from_scratch_lr=4' --mode='test'
```

Final acc: $92.2\pm0.8$%。

The trianed model is too big, so I didn't put it on github.

For more information, see my report.

