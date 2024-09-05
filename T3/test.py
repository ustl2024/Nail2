from net import *
from nail2 import *
from train import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
from skimage.transform import resize
def predict(model, image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output).cpu().squeeze().numpy()
    return output

def compute_metrics(pred, gt):
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    iou = jaccard_score(gt.flatten(), pred.flatten(), average='binary')
    dice = f1_score(gt.flatten(), pred.flatten(), average='binary')
    return iou, dice

test_image_path = r'D:\pythonProject\T3\archive\nails_segmentation\images\7e9f5818-4425-4d8a-808a-4673d96fa250.jpg'
gt_image_path = r'D:\pythonProject\T3\archive\nails_segmentation\labels\7e9f5818-4425-4d8a-808a-4673d96fa250.jpg'
prediction = predict(model, test_image_path)

gt_image = Image.open(gt_image_path).convert('L')
gt = np.array(gt_image)
gt = resize(gt, prediction.shape, mode='constant', preserve_range=True)


iou, dice = compute_metrics(prediction, gt)
print(f"IOU: {iou:.4f}, Dice: {dice:.4f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(Image.open(test_image_path))

plt.subplot(1, 3, 2)
plt.title('Prediction')
plt.imshow(prediction, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Ground Truth')
plt.imshow(gt, cmap='gray')

plt.show()
