from transformers import CLIPProcessor, CLIPModel
from PIL import Image
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("photo_forward.jpg")

# text = ['number of lane in this image is 0', 'number of lane in this image is 1', 'number of lane in this image is 2', 'number of lane in this image is 3']
text = ['lane', 'pear', 'apple', 'flower']
# text = ['0 lane', '1 lanes', '2 lanes', '3 lanes']


inputs = processor(text= text, images=image, return_tensors='pt', padding = True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim = 1)

for i in range(len(text)):
    print(text[i], ':',probs[0][i])

# #伪代码
# #1. 提取单模态特征，image->I_i, text->T_i
# I_f = image_encoder(image) #(n,d_i)  a row
# T_f = text_encoder(text)  #(n,d_t)  a row

# #2  两个特征过多模态embadding，提取多模态特征, 同时对两个多模态特征做layer norm
# I_e = l2_normalize(np.dot(I_f,W_i),axis=1) #[n,d_i]*[d_i,d_e] = [n,d_e]
# T_e = l2_normalize(np.dot(T_f,W_t),axis=1) #[n,d_t]*[d_t,d_e] = [n,d_e]

# #3 计算图片，文字向量的余弦相似度
# logits = np.dot(I_e,T_e.T) * np.exp(t) # [n,n]

# #4.计算loss
# labels = np.arange(n)
# loss_i = cross_entropy_loss(logits, labels, axis=0)
# loss_t = cross_entropy_loss(logits, labels, axis=1)
# loss = (loss_i + loss_t)/2 
