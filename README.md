# Fourier Neural Operators: Hyperparameter Optimization and Autoregressive Time Step Prediction

## Project for ENM 5310 Data-driven Modeling and Probabilistic Scientific Computing Fall 2024

### Shawn Koohy<sup>1</sup>

## Abstract: 
<em>Fourier Neural Operators (FNOs) are a powerful tool for learning mappings between infinite-dimensional function spaces, offering efficiency and resolution invariance for solving partial differential equations (PDEs). This paper explores the application of FNOs on two benchmark problems: the Burgers' equation and the Korteweg–De Vries (KdV) equation. For the Burgers' equation, we conduct an extensive hyperparameter optimization study to minimize the $L^2$ error and loss allowing the model to achieve high accuracy across all examples. For the KdV equation, we extend the FNO framework to an autoregressive setting, predicting multiple future time steps based on previous predictions. Our results demonstrate the effectiveness and expressivity of FNOs, achieving an average $L^2$ error of $0.0016$ for Burgers' and $0.08$ for KdV. This showcases the potential of FNOs to solve complex PDE systems in a wide range of applications..</em>

</sub></sub><sub>1</sup> Department of Mechanical Engineering and Applied Mechanics University of Pennsylvania Philadelphia, PA 19014, USA</sub></sub><br>

```python main.py --config=configs/defaults.py``` or ```python sweep.py --config=configs/defaults.py```

[Google Drive link to datasets](https://drive.google.com/drive/folders/1Ai-yIE6tq2jVuobfPCK5T_UU8S07jVuS?usp=sharing)
