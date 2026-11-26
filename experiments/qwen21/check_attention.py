"""Compare the experimental processor to dense logits on a tiny real Qwen attention layer."""
import torch
from torch.nn.functional import scaled_dot_product_attention
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Attention, QwenImage21KVLayerCache, QwenImage21AttnProcessor, _qwenimage21_prepare_qkv
from attention import RoleAttention

torch.manual_seed(1)
attn=QwenImage21Attention(dim=16,heads=2,dim_head=8).eval()
refs=[torch.tensor([1,0]), torch.tensor([0,1,1])]
targets=[torch.tensor([1,0,0,1]), torch.tensor([0,1,0,1])]
segs=[(0,2,True),(2,4,False),(4,5,True),(5,8,False)]
prefill=torch.randn(1,12,16)
pad=torch.ones(1,12,dtype=torch.bool)
pad[:,1]=False
def dense(h, cache, mode):
 q,k,v,_=_qwenimage21_prepare_qkv(attn,h,None,cache,mode,None)
 bias=torch.zeros(1,1,4,12)
 bias[:,:,0,2]=2
 bias[:,:,0,6:8]=-4
 bias[:,:,1,2]=-4
 bias[:,:,1,6:8]=2
 bias[:,:,:,1]=float('-inf')
 y=scaled_dot_product_attention(q[:,-4:].transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=bias)
 return attn.to_out[1](attn.to_out[0](y.transpose(1,2).flatten(2)))
with torch.no_grad():
 p=RoleAttention(refs,targets)
 cache=QwenImage21KVLayerCache()
 native=QwenImage21AttnProcessor()(attn,prefill,segments=segs,key_valid=pad)
 result=p(attn,prefill,layer_cache=cache,kv_cache_mode='extract',cache_write_slice=slice(0,8),segments=segs,key_valid=pad)
 torch.testing.assert_close(result[:,:8],native[:,:8],rtol=0,atol=0)
 torch.testing.assert_close(result[:,8:],dense(prefill,None,None))
 h=torch.randn(1,4,16)
 result=p(attn,h,attention_mask=pad[:,None,None,:],layer_cache=cache,kv_cache_mode='cached')
 torch.testing.assert_close(result,dense(h,cache,'cached'))
 print('Native prefix identical; role, background, overlap, padding and cached output match dense attention')
