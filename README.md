
## 🔬Environment
See requirements.txt

The version of Python is 3.10



## ⚙Reverse-engineering
For example, to conduct reverse-engineering for the embedding on the Flickr dataset:
```bash
cd embedding
python run_write_re_loss_to_txt_initial.py --gpu 0 --filePath ./flickr30k/ --model_type sd \
--num_iter 1000 --write_path ./result
```
reconstructed embeddings will be saved in "--write_path"
## embedding-to-text
### 1. train an E2T model for CLIP
Following [vec2text](https://github.com/vec2text/vec2text)

Our pre-trained model is coming soon.

### 2. invert embedding to text
Set the path in inverse_embedding.py 
```python
inversion_model_path = "path to your zero model"
corrector_model_path = "path to your correction model"
model_id = "stable-diffusion-v1-5(your local path)"
embedding_path = "path to your reconstructed embeddings folder"
origin_image_path = "path to your target images"
```

and then run:
```bash
cd vec2text
python inverse_embedding.py 
```
you can find the result in "result.json" 

