# ZeroNAS
The code for ZeroNAS: Differentiable Generative Adversarial Networks Search for Zero-Shot Learning.
## Requirements:
```
python==3.6, pytorch==0.3.1 and other dependencies in requirements.txt
```

Note: The Graphviz software is required before installing the graphviz python package through ```sudo apt-get install graphviz```.
## Datasets:
The code uses the ResNet101 features and seen/unseen splits provided by the paper: Feature Generating Networks for Zero-Shot Learning.
The features of CUB, AWA, FLO and SUN dataset can be download here: <http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip>
## Running instructions:
- To carry out architecture search, run ```scripts.sh``` with all the parameters set properly. This code will generate the output files regarding to the searched architectures to ```./output``` directory.
Note that the validation performance in this step does not indicate the final performance of the architecture. 

- To evaluate the searched architectures, one must train the architecture from scratch using ```clswgan_retrain.py```. Notably, the ```genotype_G``` and ```genotype_G``` variable need to set as the obtained generator and discriminator architecture respectively. The trained models for each dataset are saved to ```./trained_models``` directory, which is further used by ```generate_feature.py``` to synthesize unseen class features for each dataset.
## Visualization:
The learned architectures for generator and discriminator can be visualized through ```plot.py```. (Package graphviz is required to visualize the learned architectures.)

For example:
```
python plot.py "[('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_relu', 6), ('fc_relu', 5)]" "[('fc_lrelu', 2), ('fc_lrelu', 0), ('fc_lrelu', 2), ('fc_relu', 1), ('fc_lrelu', 4), ('fc_relu', 2), ('fc_lrelu', 5), ('fc_relu', 3), ('fc_lrelu', 6), ('fc_lrelu', 5)]"
```
