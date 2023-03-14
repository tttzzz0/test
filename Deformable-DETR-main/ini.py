import torch
pretrained_weights = torch.load(r'E:\PycharmProjects\Deformable-DETR-main\Deformable-DETR-main\r50_deformable_detr-checkpoint.pth')
num_class = 5 # 自己数据集分类数
pretrained_weights['model']['class_embed.0.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.0.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.1.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.1.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.2.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.2.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.3.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.3.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.4.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.4.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.5.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.5.bias'].resize_(num_class+1)
pretrained_weights['model']['query_embed.weight'].resize_(100, 512) # 此处50对应生成queries的数量，根据main.py中--num_queries数量修改
torch.save(pretrained_weights, 'de_detr-r50_%d.pth'%num_class)
