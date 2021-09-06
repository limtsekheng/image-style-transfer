import PIL
import numpy as np
import torch
import torchvision
import torch.autograd as autograd
import torchvision.transforms as T
import torch.nn.functional as F

class StyleTransfer():
	
	def __init__(self):
		self.model = torchvision.models.squeezenet1_1(pretrained=True).to('cpu')
		for param in self.model.parameters():
			param.requires_grad = False
		self.IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
		self.IMAGENET_STD = np.array([0.229, 0.224, 0.225])


	def content_loss(self, content_weight, content_current, content_original):
		loss = content_weight * (torch.sum((content_original - content_current)**2))
		return loss


	def gram_matrix(self, features):
		_, C, H, W = features.size()
		Fl = features.view(C, H*W)
		gram = torch.mm(Fl, Fl.t())
		gram = gram/(H*W*C)
		return gram


	def style_loss(self, feats, style_layers, style_targets, style_weights):
	    style_loss = 0
	    i = 0
	    for n in style_layers:
	        current_gram = self.gram_matrix(feats[n])         
	        style_loss = style_loss + style_weights[i] * torch.sum((current_gram - style_targets[i])**2)
	        i+=1      
	    return style_loss


	def get_features(self, image, model):
	    features = []
	    for name,layer in (model._modules.items()):
	        image = layer(image)
	        features.append(image)        
	    return features


	def preprocess(self, img, size=(224, 224)):
	    transform = T.Compose([
	        T.Resize(size),
	        T.ToTensor(),
	        T.Normalize(mean=self.IMAGENET_MEAN.tolist(), std=self.IMAGENET_STD.tolist()),
	        T.Lambda(lambda x: x[None]),
	    ])
	    return transform(img)


	def deprocess(self, img, should_rescale=True):
	    transform = T.Compose([
	        T.Lambda(lambda x: x[0]),
	        T.Normalize(mean=[0, 0, 0], std=(1.0 / self.IMAGENET_STD).tolist()),
	        T.Normalize(mean=(-self.IMAGENET_MEAN).tolist(), std=[1, 1, 1]),
	        T.Lambda(self.rescale) if should_rescale else T.Lambda(lambda x: x),
	        T.ToPILImage(),
	    ])
	    return transform(img)


	def rescale(self, x):
	    low, high = x.min(), x.max()
	    x_rescaled = (x - low) / (high - low)
	    return x_rescaled


	def style_transfer(self, content_image, style_image, content_layer=5, content_weight=1e-2, style_layers=(1,3,6,7), style_weights=(30000,100,10,1), max_iter=200):

	    # content_img = self.preprocess(PIL.Image.open(content_image))
	    content_img = self.preprocess(content_image)
	    features = self.get_features(content_img, self.model.features)
	    content_target = features[content_layer].clone()

	    style_img = self.preprocess(PIL.Image.open(style_image))
	    features = self.get_features(style_img, self.model.features)   
	    style_targets = []
	    for i in style_layers:
	        style_targets.append(self.gram_matrix(features[i].clone()))

	    img = content_img.clone().detach()

	    img_var = autograd.Variable(img,  requires_grad=True)

	    optimizer = torch.optim.Adam([img_var], lr=2)
	    
	    for i in range(max_iter):
	            if i< max_iter-10:
	                img.clamp_(-1.5, 1.5)
	            optimizer.zero_grad()

	            feats = self.get_features(img_var, self.model.features)

	            c_loss = self.content_loss(content_weight, feats[content_layer], content_target)
	            s_loss = self.style_loss(feats, style_layers, style_targets, style_weights)
	            loss = c_loss + s_loss 

	            loss.backward(retain_graph=True)          
	            if i == max_iter-20:
	                optimizer = torch.optim.Adam([img_var], lr=0.1) 
	            optimizer.step()

	    return self.deprocess(img)


	def obtain_result(self, img, style_path):
		res = self.style_transfer(img, style_path)
		content = self.deprocess(self.preprocess(img))
		return content, res



