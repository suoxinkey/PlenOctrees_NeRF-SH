import torch



def vis_density(model, L= 32):
    x = torch.linspace(0,1,steps=L).cuda()
    grid_x ,grid_y,grid_z = torch.meshgrid(x, x,x)
    xyz = torch.stack([grid_x ,grid_y,grid_z], dim = -1)  #(L,L,L,3)

    xyz = xyz * (model.maxs-model.mins) + model.mins

    xyz = xyz.reshape((-1,3)) #(L*L*L,3)

    _,density = model.spacenet(xyz, None, model.maxs, model.mins) #(L*L*L,1)

    density = torch.nn.functional.relu(density)
    density = density / density.max()
    xyz = xyz[density.squeeze()>0.3,:]
    density = density[density.squeeze()>0.3,:].repeat(1,3)

    density[:,1:3] = 0

    return xyz.unsqueeze(0), density.unsqueeze(0)
    






