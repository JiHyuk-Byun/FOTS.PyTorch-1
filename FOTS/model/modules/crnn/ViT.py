
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


# Conv2d를 이용해서먼저 convolution을 해준다음에 쪼개줌. performance측면에서 1번보다 나음
class PatchEmbedding(nn.Module):
  def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size = 224):
    self.patch_size = patch_size
    super().__init__()
    self.projection = nn.Sequential(
    #stride size = kernel size=patch size로 설정해서 patch에 linear적용하는거랑 똑같게 해줌, emb_size=16*16*3
      nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
      Rearrange('b e (h) (w) -> b (h w) e')
    )
    ## 아래는 token embedding과정. 이때 cls, pos token은 하나의 patch의 embedding size와 같도록 해줌

    # 최종 L 번째 Layer의 0 번째 patch인 class token생성, BERT처럼 맨앞에다가 학습가능한 token을 부착해줌. 결국 해당 이미지의 클래스를 classification하기 위함
    self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
    #각각의 patch별로 patch내 embedding들에 더해줄 position token 생성. +1은 cls_token
    self.positions = nn.Parameter(torch.randn(((img_size // patch_size)**2 + 1, emb_size)))

  def forward(self, x: Tensor) -> Tensor:
    b, _, _, _ = x.shape
    x = self.projection(x)
    cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) #batch size개수만큼 늘려줌
    x = torch.cat([cls_tokens, x], dim=1) #class token 이제 부착하자
    x += self.positions
    return x


## MHA
class MultiHeadAttention(nn.Module):
  #heads는 concatenate할 K,Q,V 연산결과로 나오는 attention head들의 개수
  def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
    super().__init__()
    #key, query, value 생성. 이때 이값들은 모두 input으로부터 생성되는 학습가능한 값
    self.emb_size = emb_size
    self.num_heads = num_heads
    self.qkv = nn.Linear(emb_size, emb_size * 3)
    self.att_drop = nn.Dropout(dropout)
    self.projection = nn.Linear(emb_size, emb_size)
    self.scaling = (self.emb_size // num_heads) ** -0.5

  def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
    #각각의 patch 들을 8조각내서 모든 patch들을 한번에 같은 embedding 순서끼리 묶는듯. 
    #패치별로 key query value를 생성하는줄 알았는데 그게 아닌가봐, 이미지 전체에서 참조해서 생성하는듯
    qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3) #내가 만들려는 attention head의 개수는 당연히 KQV짝들의 개수와 같아야겠지 
    queries, keys, values = qkv[0], qkv[1], qkv[2]

    #einsum에서 식을 살펴보면 keys가 transpose된 다음에 queries가 곱해지는걸 확인할 수 있다. 
    #즉, dot product수행=cosine 유사도구하는과정
    energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 

    #mask할 부분은 -무한대로 채워넣음
    if mask is not None:
      fill_value = torch.finfo(torch.float32).min #torch.float32에서 가장 작은 숫자 반환
      energy.mask_fill(~mask, fill_value)

    #식 그대로 구현
    att = F.softmax(energy, dim=-1) * self.scaling 
    att = self.att_drop(att)

    #softmax를 통해 구한 KV값에 value elementwise 곱해서 최종 encoding값 구함
    #다시 rearrange이용해서 원래 flatten patch 사이즈로 변환
    out = torch.einsum('bhal, bhlv -> bhav', att, values) 
    out = rearrange(out, "b h n d -> b n (h d)")
    out = self.projection(out)
    return out


##Residual Block
class ResidualAdd(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn
  
  def forward(self, x, **kwargs):
    res = x
    x = self.fn(x, **kwargs)
    x += res
    return x


## MHA이후 MLP 구현. 객체지향적으로 nn.Sequential 상속받아서 구현. 구조는 Linear로 사이즈 증가한후 다시줄이는 구조, GELU이용
class FeedForwardBlock(nn.Sequential):
  def __init(self, emb_size: int, expansion: int = 4, drop_p: float = 0.): 
    super().__init__(
        nn.Linear(emb_size, expansion * emb_size),
        nn.GELU(),
        nn.Dropout(drop_p),
        nn.Linear(expansion * emb_size, emb_size),
    )

## 이제 layer별로 정의된 layer class들을 nn.Sequential상속을 통해 합쳐서 Encoder 생성해주자
class TransformerEncoderBlock(nn.Sequential):
  def __init__(self,
               emb_size: int = 768,
               drop_p: float = 0.,
               forward_expansion: int = 4,
               forward_drop_p: float = 0.,
               ** kwargs):
    super().__init__(
        ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(emb_size, **kwargs),
            nn.Dropout(drop_p)
        )),
        ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(
                emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p)
        )
        ))


##Transfromer Encoder Block이 depth번 반복되는 전체 Encoder
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


## Classification Head, 각각의 multihead들의 평균값을 구해주어 하나의 head로 추린뒤 linear에 돌려준다.
class ClassificationHead(nn.Sequential):
  def __init__(self, emb_size: int = 768, n_classes: int = 1000):
    super().__init__(
        Reduce('b n e -> b e', reduction='mean'), # einops.reduce, reduction. batch의 평균값 구하기
        nn.LayerNorm(emb_size),
        nn.Linear(emb_size, n_classes))

class ViT(nn.Sequential):
  def __init__(self,
               in_channels: int = 3,
               patch_size: int = 16,
               emb_size: int = 768,
               img_size: int = 224,
               depth: int = 12,
               n_classes: int = 1000,
               **kwargs):
    super().__init__(
        PatchEmbedding(in_channels, patch_size, emb_size, img_size),
        TransformerEncoder(depth, emb_size=emb_size, **kwargs),
        ClassificationHead(emb_size, n_classes)
    )


# summary(ViT(), (3, 224, 224), device='cpu')