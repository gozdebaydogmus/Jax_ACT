# sanity_shapes.py
import jax, jax.numpy as jnp
from act_model_flax_vae import ACTVAEModel  # senin model sınıfı adıyla değiştir
from dataloader_jax import make_batches         # senin batch fonksiyonunla değiştir

rng = jax.random.PRNGKey(0)
batch = next(make_batches())  # {'obs':..., 'action':..., ...} bekleniyor
print("obs keys:", list(batch['obs'].keys()) if isinstance(batch['obs'], dict) else type(batch['obs']))
print("obs shape:", {k: v.shape for k,v in batch['obs'].items()} if isinstance(batch['obs'], dict) else batch['obs'].shape)
print("action shape:", batch['action'].shape)

model = ACTModel(...)      # config’ine göre doldur
variables = model.init({'params': rng}, batch)  # veya {'params': rng, 'latent': rng} kullanıyorsan
out = model.apply(variables, batch, mutable=False)
print("forward ok. out keys:", out.keys() if hasattr(out, 'keys') else type(out))
