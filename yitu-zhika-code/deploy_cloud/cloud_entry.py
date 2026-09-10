import os
for name in ('INFERENCE_API_KEY', 'ADMIN_API_KEY', 'MERCHANT_API_KEY'):
    if len(os.environ.get(name, '')) < 24 or os.environ[name].startswith('dev-'):
        raise RuntimeError(f'Set a strong {name} in cloud environment variables')
from app.inference_service import app, get_pipeline
pipeline = get_pipeline()
pipeline.load_macros_if_ready()
pipeline.load_category_if_ready()
