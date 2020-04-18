import torch


def batchify_ray(model, rays, bboxes, chuncks = 1024*7):
    N = rays.size(0)
    if N <chuncks:
        return model(rays, bboxes)

    else:
        rays = rays.split(chuncks, dim=0)
        bboxes = bboxes.split(chuncks, dim=0)

        colors = [[],[]]
        depths = [ [],[]]
        acc_maps = [ [],[]  ]

        for i in range(len(rays)):
            stage2, stage1 = model( rays[i], bboxes[i])
            colors[0].append(stage1[0])
            depths[0].append(stage1[1])
            acc_maps[0].append(stage1[2])

            colors[1].append(stage2[0])
            depths[1].append(stage2[1])
            acc_maps[1].append(stage2[2])

        colors[0] = torch.cat(colors[0], dim=0)
        depths[0] = torch.cat(depths[0], dim=0)
        acc_maps[0] = torch.cat(acc_maps[0], dim=0)

        colors[1] = torch.cat(colors[1], dim=0)
        depths[1] = torch.cat(depths[1], dim=0)
        acc_maps[1] = torch.cat(acc_maps[1], dim=0)

        return (colors[1], depths[1], acc_maps[1]), (colors[0], depths[0], acc_maps[0])
