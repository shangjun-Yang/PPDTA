# PPDTA
PPDTA: Predicting Drug-Target Binding Affinity Using Pharmacophore Fingerprints and PPI Network

## Model Architecture 

![image-20240813200325041](./image.png)

## Requirements

- The most important python packages are：
- einops==0.8.0
- matplotlib==3.6.2
- networkx==3.1
- numpy==1.20.3
- pandas==1.5.2
- prefetch_generator==1.0.3
- scikit_learn==1.2.2
- torch==1.12.1
- torch_geometric==2.5.3
- tqdm==4.66.5


For using our model more conveniently, we provide the requirements file <requirements .txt>  to install environment directly.

```python
pip install requirements.txt
```

## Dataset

We use three datasets, i.e. Davis, KIBA and BindingDB_Kd datasets.

## Example usage

1. Preparing Data 

   ```python
   python data_prepare.py Davis
   ```

2. Training the model

   ```python
   python main.py Davis
   ```

3. Testing  the model 

   ```python
   python test_model.py Davis
   ```

## Contact

If you have any questions, please feel free to contact Jiao Wang (Email: wangjiao@mail.imu.edu.cn)


