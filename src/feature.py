from torchvision import transforms, models

def featureExtractor(im):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    im = transform(im)
    im = im.unsqueeze(0)
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()
    return alexnet(im)
