# ZeroNAS
The code for ZeroNAS: Differentiable Generative Adversarial Networks Search for Zero-Shot Learning.
## Requirements:
python==3.6, pytorch==0.3.1 and other dependencies in requirements.txt

Note: Package graphviz is required to visualize the learned cells.
## Datasets:
The code uses the ResNet101 features and seen/unseen splits provided by the paper: Feature Generating Networks for Zero-Shot Learning.
The features can be download here: http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip
## Architecture Search and Evaluation:
## Visualization:
The learned architectures for generator and discriminator can be visualized through plot.py. 

For example:

python plot.py "[('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_lrelu', 2), ('fc_lrelu', 1), ('fc_relu', 6), ('fc_relu', 5)]" "[('fc_lrelu', 2), ('fc_lrelu', 0), ('fc_lrelu', 2), ('fc_relu', 1), ('fc_lrelu', 4), ('fc_relu', 2), ('fc_lrelu', 5), ('fc_relu', 3), ('fc_lrelu', 6), ('fc_lrelu', 5)]"
