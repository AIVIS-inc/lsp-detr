---
license: apache-2.0
library_name: transformers
---


```python
from transformers import AutoModelForObjectDetection, AutoImageProcessor


processor = AutoImageProcessor.from_pretrained(
    "RationAI/LSP-DETR", trust_remote_code=True
)
model = AutoModelForObjectDetection.from_pretrained(
    "RationAI/LSP-DETR", trust_remote_code=True
)

inputs = processor(img, device=device, return_tensors="pt")
outputs = model(**inputs)
results = processor.post_process(outputs)
results = processor.post_process_instance(
  results, height=img.size[1], width=img.size[0]
)
```