# MaskingUNet
For paper

Researching the possible benefits of using multiplication instead of concatenation for UNet residual. Comparing UNet and MaskingUNet for 3 tasks, image segmentation, image generation, and image super resolution, using COCO17 and flowers102 datasets.







Building the docker image
> docker build . -t b-lanc/maskingunet

Running the docker container
> docker run -dit --name=MU --runtime=nvidia --gpus=all --shm-size=16gb -v /mnt/Data/datasets/coco:/datasets/coco -v /mnt/Data/datasets/102flowers:/datasets/102flowers -v /mnt/Data2/DockerVolumes/MU:/saves -v .:/workspace b-lanc/maskingunet

Going inside the container
> docker exec -it MU /bin/bash