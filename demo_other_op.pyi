import torch
import numpy as np
import cv2

data = cv2.imread("test.jpg")
# print(data)

# a = np.zeros([2, 2])
out = torch.from_numpy(data)
print(out)

out = out.to(torch.device("cuda"))
print(out.is_cuda)

out = torch.flip(out,dims=[0])

out = out.to(torch.device("cpu"))
print(out.is_cuda)

data = out.numpy()

cv2.imshow("test", data)
cv2.waitKey(0)