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

This diagram illustrates the architecture of a GPT model, showing the complete flow from input preprocessing to output probabilities. The process begins with Input Preprocessing, where the text data undergoes character-level mapping and is split into training and validation sets. The Embedding Layers follow, transforming input tokens into dense vectors and adding positional embeddings to retain sequence information. The core of the model consists of Transformer Blocks, repeated 8 times. Each block includes a Masked Multi-Head Attention mechanism (8 Attention Heads) that calculates attention scores and processes information from different positions, followed by an Add & Norm step that normalizes the data with residual connections. This is succeeded by a Feed Forward Network for further transformation, another Add & Norm layer. Finally, the Output Layer performs a linear projection of the normalized data to generate logits, which are then passed through a Softmax function to produce output probabilities, facilitating tasks like text generation and prediction during the training process.

## Results and Observations

### Sample of the Dataset the model is trained on

.The traffic moved on and a few minutes later Mr Dursley arrived in the Grunnings parking lot his mind back on drills .Mr Dursley always sat with his back to the window in his office on the ninth floor .If he hadnt he might have found it harder to concentrate on drills that morning .He didnt see the owls swooping past in broad daylight though people down in the street did they pointed and gazed openmouthed as owl after owl sped overhead .Most of them had never seen an owl even at nighttime .Mr Dursley however had a perfectly normal owlfree morning .He yelled at five different people .He made several important telephone calls and shouted a bit more .He was in a very good mood until lunchtime when he thought hed stretch his legs and walk across the road to buy himself a bun from the bakery .Hed forgotten all about the people in cloaks until he passed a group of them next to the bakers .

### Generated Text After training

 They all got to their feet and stick as the tiny moan both saying the vanish wand looking at her .Harry wondered whether how torn Buckbeak more heads were there and even sat down Hermione rolled up the pitch besiding Rons wands .The Bull followed himself onto the table and they glared around and divided us at eleven Hedwig Fudge muttered .Oh he got care now then he said to Umbridge to the door and he interrogated her .

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

### Conclusion

In this experiment, we successfully created a GPT model from scratch, featuring 8 heads and 8 layers, to perform text generation based on the Harry Potter dataset. We further enhanced the model's performance using high-performance computing techniques. 

To conclude, while the final generated text exhibits the stylistic elements of the Harry Potter series and demonstrates syntactical coherence in several parts, there remain areas for improvement. For future work, we propose training the model on the OpenWebText dataset, followed by fine-tuning on the Harry Potter dataset, to produce more contextually appropriate responses.

