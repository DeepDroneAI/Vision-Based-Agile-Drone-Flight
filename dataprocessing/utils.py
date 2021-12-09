import torch

def weighted_mse_loss(outputs, target, weight1, weight2, weight3):
    
    o = torch.sum(weight1 * (outputs[:,0] - target[:,0]) ** 2)
    u = torch.sum(weight2 * (outputs[:,1:3] - target[:,1:3]) ** 2)
    t = torch.sum(weight3 * (outputs[:,3] - target[:,3]  ) ** 2)

    return o + u + t

