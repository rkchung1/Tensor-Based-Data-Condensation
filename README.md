# Tensor-Based-Data-Condensation
This project explores data condensation for tabular datasets using tensor methods. Data condensation is the process of generating a smaller (often synthetic) dataset that maintains model performance. Our goal is to reduce data size while preserving essential information for training and downstream tasks.


## to-dos's - Bela
1. Redo pipeline for covertype, running with auto encoder
2. Fix Tucker code so it can compress "less," look into tensorly library and try to make it work with the GPU or try to run it locally
3. Make parameter selection doable / easy.
4. Analyze steps of the compression / make heat maps that will show which areas can be compressed more without compromising performance
5. Homogenize pipeline.
6. Add script that will determine whether or not the dataset needs to be auto encoded.
