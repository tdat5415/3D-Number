# 3D-Number
Hackathon - Dacon

---

안녕하세요. 낼모레입니다.
Dacon 월간 챌린지 숫자 3D 이미지 분류에서
Private 11위(0.95371)를 달성했습니다.

CurriculaFace를 활용한 Xception 모델을 사용하였고
전처리 기법은 3D 회전 Augmentation후 정사영하는 방식을 이용했습니다.

링크 : <https://dacon.io/competitions/official/235951/codeshare/6583?page=1&dtype=recent>

---

##### 주요코드 부가 설명

---

```python
def dots_to_img24(dots, img_size=46):
    imgs = []
    shadow_xyz = dots[:, [[0,1], [0,2], [1,2]]]
    shadow_xyz = np.transpose(shadow_xyz, (1, 0, 2)) # (3, n, 2)
    shadow_xyz = (shadow_xyz+0.75)/1.5
    shadow_xyz = np.round(shadow_xyz*img_size)
    shadow_xyz = np.clip(shadow_xyz, 0, img_size-1).astype(np.int32)
    for shadow in shadow_xyz:
        board = np.zeros((img_size, img_size), dtype=np.float32)
        board[shadow[:,0], shadow[:,1]] = 1.
        board_flip = board[:,::-1]
        for i in range(4):
            imgs.append(np.rot90(board, k=i))
            imgs.append(np.rot90(board_flip, k=i))
    return imgs

def img24_to_grid_img(img24, img_size=46):
    board = np.zeros((img_size*5+16, img_size*5+16), dtype=np.float32)
    for i, img in enumerate(img24):
        cur_x = (i%5)*(img_size+4)
        cur_y = (i//5)*(img_size+4)
        board[cur_y:cur_y+img_size, cur_x:cur_x+img_size] = img
        
    board = np.pad(board, ((5,5),(5,5)), 'constant', constant_values=0)
    return board
```

xyz좌표기반 점 데이터를 정사영하여 이미지화를 하는 방법입니다.

일단 좌표를 0~1로 정규화 할 필요가 있었습니다.

마침 좌표들이 대부분 -0.75~0.75 사이값들 뿐이라서 정규화하기 수월했습니다.

그 뒤 0~1사이 값들을 0~46 으로 다시 정규화합니다. (46은 여기서 이미지 한변의 사이즈입니다.)

0~46 값들을 반올림후 그 값을 인덱스처럼 활용하여 빈 공백이미지에 채워주면 됩니다.

---

```python
import numpy as np

def rotate(a, b, c, dots):
    mx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    my = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    mz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    m = np.dot(np.dot(mx,my),mz)
    dots = np.dot(dots, m.T)
    return dots
```

![191392909-fd233655-e25f-45eb-86d6-e6049d128c2c](https://user-images.githubusercontent.com/48349693/192729302-68edd9b4-85a4-47ad-a9b2-36f218d1e04e.png)

이 공식은 원점 중심으로 좌표들을 회전시키는 공식입니다.

그 공식을 코드화 했습니다.

먼저 세 공식의 dot-product를 구하고(M) 데이터에 다시 dot-product를 합니다.

M을 transpose한 이유는 dots의 axis가 반대(n,3)이기 때문입니다. 그래서 연산순서도 바꿔야되요.

~~이 코드를 초보분들을 위해 코드공유 게시판에 공유했었는데 정작 상위권 10팀중 4팀이 이 코드 사용했네요.. 패착입니다..~~

---

```python
class CosSimLayer(tf.keras.layers.Layer):
    def __init__(self, n_classes, regularizer=None, name='CosSimLayer', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.regularizer = regularizer

    def get_config(self):
        config = super().get_config()
        config.update({"n_classes":self.n_classes, "regularizer":self.regularizer})
        return config
    
    def build(self, embedding_shape):
        self._w = self.add_weight(shape=(embedding_shape[-1], self.n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self.regularizer,
                                  name='cosine_weights')

    def call(self, embedding, training=None):
        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits') # (batch, 4096)
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights') # (4096, 10)
        cosine_sim = tf.matmul(x, w, name='cosine_similarity') # (batch, 10)
        return cosine_sim
```
입력받은 embedding값과 이 레이어의 weight와의 코사인 유사도를 리턴합니다.

이해할려면 내적공식을 알고있어야 합니다. ( 내적 공식 : |A||B|cosΘ = A·B = a1b1 + a2b2 + ... )

내적공식에서 cosΘ 만을 구하기 위해 각 |A|,|B|는 1이어야 하므로 l2_normalize하면 됩니다.

---

```python
class CurricularFaceLoss(tf.keras.losses.Loss):
    def __init__(self, scale=30, margin=0.5, alpha=0.99, name="CurricularFaceLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = scale
        self.margin = margin
        self.alpha = alpha
        self.t_value = tf.Variable(0., name='t-value')
        self.eps = 1e-7
        self.training = True
    
    def positive_forward(self, y_logit):
        cosine_sim = y_logit 
        theta_margin = tf.math.acos(cosine_sim) + self.margin
        y_logit_pos = tf.math.cos(theta_margin)
        return y_logit_pos
    
    def negative_forward(self, y_logit_pos_masked, y_logit):
        hard_sample_mask = y_logit_pos_masked < y_logit # (N, n_classes)
        y_logit_neg = tf.where(hard_sample_mask, tf.square(y_logit)+self.t_value*y_logit, y_logit)
        return y_logit_neg
    
    def forward(self, y_true, y_logit):
        y_logit = tf.clip_by_value(y_logit, -1.0+self.eps, 1.0-self.eps)
        y_logit_masked = tf.expand_dims(tf.reduce_sum(y_true*y_logit, axis=1), axis=1) # (N, 1)
        y_logit_pos_masked = self.positive_forward(y_logit_masked) # (N, 1)
        y_logit_neg = self.negative_forward(y_logit_pos_masked, y_logit) # (N, n_classes)
        # update t
        if self.training:
            r = tf.reduce_mean(y_logit_pos_masked)
            self.t_value.assign(self.alpha*r + (1-self.alpha)*self.t_value)
        
        y_true = tf.cast(y_true, dtype=tf.bool)
        return tf.where(y_true, y_logit_pos_masked, y_logit_neg)
    
    def call(self, y_true, y_logit): # shape(N, n_classes)
        y_logit_fixed = self.forward(y_true, y_logit)
        loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_logit_fixed*self.scale)
        loss = tf.reduce_mean(loss)
        return loss
```

t-value는 적응형 변수로 positive값이 클 수록 즉, 모델이 좋아질 수록 값이 조금씩 커집니다.

그러면 hard sample을 좀 더 학습하라고 loss를 늘려줍니다.

참고 : <https://emkademy.medium.com/angular-margin-losses-for-representative-embeddings-training-arcface-2018-vs-mv-arc-softmax-96b54bcd030b>

---

```python

```




