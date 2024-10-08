---
base_model: sentence-transformers/paraphrase-MiniLM-L6-v2
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4059
- loss:CosineSimilarityLoss
widget:
- source_sentence: Looking for a pasta dish with marinara sauce and meatballs.
  sentences:
  - Chili Cheese Fries - Crispy fries topped with chili, melted cheese, and jalapenos.
    $8.99
  - Vanilla Ice Cream Sundae - Vanilla ice cream topped with whipped cream and chocolate
    syrup. $4.49
  - Spaghetti and Meatballs - Spaghetti served with marinara sauce and Italian meatballs.
    $10.99
- source_sentence: Craving a chocolate dessert with ice cream.
  sentences:
  - Buffalo Chicken Wings - Fried chicken wings tossed in buffalo sauce, served with
    ranch. $8.99
  - Chicken Caesar Wrap - Grilled chicken, romaine lettuce, Parmesan, and Caesar dressing
    in a wrap. $8.99
  - Brownie Sundae - A warm brownie topped with vanilla ice cream. $6.99
- source_sentence: Looking for a wrap with grilled veggies and hummus.
  sentences:
  - Strawberry Milkshake - A thick milkshake made with fresh strawberries and ice
    cream. $4.49
  - Grilled Veggie Wrap - Grilled vegetables with hummus wrapped in a whole wheat
    tortilla. $7.99
  - BBQ Chicken Pizza - Grilled chicken with BBQ sauce and melted cheese on a pizza
    crust. $11.99
- source_sentence: Craving a sandwich with grilled cheese and tomato soup.
  sentences:
  - Grilled Cheese with Tomato Soup - A classic grilled cheese sandwich served with
    a side of tomato soup. $7.99
  - Caramel Macchiato - A creamy espresso drink with caramel drizzle. $4.99
  - Chocolate Fudge Brownie - A rich and fudgy chocolate brownie. $5.99
- source_sentence: Iâ€™m in the mood for a light fruit salad.
  sentences:
  - Peach Sorbet - A light and refreshing peach-flavored sorbet. $3.49
  - Chicken Wings - Fried wings with buffalo sauce. $7.99
  - Citrus Mint Cooler - A refreshing blend of citrus and mint. $3.99
---

# SentenceTransformer based on sentence-transformers/paraphrase-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) <!-- at revision 3bf4ae7445aa77c8daaef06518dd78baffff53c9 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Iâ€™m in the mood for a light fruit salad.',
    'Chicken Wings - Fried wings with buffalo sauce. $7.99',
    'Citrus Mint Cooler - A refreshing blend of citrus and mint. $3.99',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 4,059 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                           |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                           |
  | details | <ul><li>min: 7 tokens</li><li>mean: 11.72 tokens</li><li>max: 19 tokens</li></ul> | <ul><li>min: 14 tokens</li><li>mean: 23.8 tokens</li><li>max: 57 tokens</li></ul> | <ul><li>min: 0.02</li><li>mean: 0.59</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                            | sentence_1                                                                                        | label             |
  |:------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:------------------|
  | <code>Do you have something cheesy and savory?</code> | <code>Cheesy Potato Skins - Potato skins topped with melted cheese and bacon. $7.99</code>        | <code>0.9</code>  |
  | <code>I want a hearty sandwich.</code>                | <code>Philly Cheesesteak - Thinly sliced steak with melted cheese on a hoagie roll. $10.99</code> | <code>0.86</code> |
  | <code>Can I get a gluten-free sandwich</code>         | <code>Mushroom Risotto - Creamy risotto with sautÃ©ed mushrooms and Parmesan. $11.99</code>       | <code>0.13</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 1
- `per_device_eval_batch_size`: 1
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 1
- `per_device_eval_batch_size`: 1
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.1232 | 500   | 0.0339        |
| 0.2464 | 1000  | 0.0288        |
| 0.3695 | 1500  | 0.0254        |
| 0.4927 | 2000  | 0.0229        |
| 0.6159 | 2500  | 0.0227        |
| 0.7391 | 3000  | 0.0218        |
| 0.8623 | 3500  | 0.0227        |
| 0.9855 | 4000  | 0.0198        |
| 1.1086 | 4500  | 0.0147        |
| 1.2318 | 5000  | 0.0179        |
| 1.3550 | 5500  | 0.015         |
| 1.4782 | 6000  | 0.0137        |
| 1.6014 | 6500  | 0.0143        |
| 1.7246 | 7000  | 0.0153        |
| 1.8477 | 7500  | 0.0154        |
| 1.9709 | 8000  | 0.0148        |
| 2.0941 | 8500  | 0.0117        |
| 2.2173 | 9000  | 0.0122        |
| 2.3405 | 9500  | 0.0118        |
| 2.4637 | 10000 | 0.0106        |
| 2.5868 | 10500 | 0.0113        |
| 2.7100 | 11000 | 0.0117        |
| 2.8332 | 11500 | 0.0114        |
| 2.9564 | 12000 | 0.0113        |
| 3.0796 | 12500 | 0.0095        |
| 3.2028 | 13000 | 0.0087        |
| 3.3259 | 13500 | 0.0106        |
| 3.4491 | 14000 | 0.0084        |
| 3.5723 | 14500 | 0.0086        |
| 3.6955 | 15000 | 0.0096        |
| 3.8187 | 15500 | 0.0099        |
| 3.9419 | 16000 | 0.0099        |


### Framework Versions
- Python: 3.12.6
- Sentence Transformers: 3.1.1
- Transformers: 4.45.1
- PyTorch: 2.4.1+cpu
- Accelerate: 0.34.2
- Datasets: 3.0.1
- Tokenizers: 0.20.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->