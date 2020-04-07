import torch


def batchify_ray(model, rays, bboxes, chuncks = 1024*13):
    N = rays.size(0)
    if N <chuncks:
        return model(rays, bboxes)

    else:
        rays = rays.split(chuncks, dim=0)
        bboxes = bboxes.split(chuncks, dim=0)

        colors =[]
        depths = []

        for i in range(len(rays)):
            color, depth = model( rays[i], bboxes[i])
            colors.append(color)
            depths.append(depth)

        colors = torch.cat(colors, dim=0)
        depths = torch.cat(depths, dim=0)


        return colors, depths
