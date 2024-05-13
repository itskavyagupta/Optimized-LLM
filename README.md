# Enhancing Large Language Model Training and Inference Using High-Performance Computing Techniques

## Description of the Project

This project aims to develop a small GPT from scratch and enhance the training and inference of the developed model using high-performance computing (HPC) techniques. Our objective is to understand and implement the GPT architecture on the Harry Potter Book dataset and improve its efficiency through various HPC methods. The ultimate goal is to create a resource-efficient model that can perform well in resource-constrained environments.

## Project Milestones

1. **Project Initialization**
   - Defined project scope and objectives.
   - Set up initial project structure and environment.

2. **Development of GPT Architecture** 
   - Developed GPT architecture from scratch.
   - Conducted extensive literature review and understanding of decoder-based models.
  
3. **Training and Evaluation**
   - Conducted multiple training iterations to optimize performance.
   - Evaluated model on the Harry Potter dataset.

3. **Profiling**
   - Implemented cProfiling to identify bottlenecks.

4. **Implementation of Data Parallelism**
   - Implemented data parallelism using PyTorch Distributed.
   - Achieved significant reduction in training time.

5. **Model Quantization**
   - Applied quantization techniques to reduce model size which reduces the Inference time.

## Model Architecture

![Copy of prog4](https://github.com/itskavyagupta/Optimized-LLM/assets/66244523/8b421f15-45cf-4c1c-834e-d3f851b83718)

## Results and Observations

### Dataset

.The traffic moved on and a few minutes later Mr Dursley arrived in the Grunnings parking lot his mind back on drills .Mr Dursley always sat with his back to the window in his office on the ninth floor .If he hadnt he might have found it harder to concentrate on drills that morning .He didnt see the owls swooping past in broad daylight though people down in the street did they pointed and gazed openmouthed as owl after owl sped overhead .Most of them had never seen an owl even at nighttime .Mr Dursley however had a perfectly normal owlfree morning .He yelled at five different people .He made several important telephone calls and shouted a bit more .He was in a very good mood until lunchtime when he thought hed stretch his legs and walk across the road to buy himself a bun from the bakery .Hed forgotten all about the people in cloaks until he passed a group of them next to the bakers .

### Generated Text

### Data Parallelism

| Configuration                | Training Time (seconds) |
|------------------------------|-------------------------|
| Baseline                     | 6365                    |
| Data Parallelism with 2 GPUs | 3817                    |
| Data Parallelism with 4 GPUs | 2754                    |

Using data parallelism, we reduced the training time by almost 57%.

### Quantization

| Model           | Size (MB) |
|-----------------|-----------|
| Baseline Model  | 169.4     |
| Quantized Model | 93.8      |

Applying quantization reduced the model size by almost 45%.

