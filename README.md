# Overview

This package provides CycleGAN and generator implementations to transform `both female and male` selfies to anime.

Many projects were made to train CycleGAN based models to transform selfie to anime. Most of the works are based on selfie2anime [dataset][selfie2anime] with high quality results.
But there is one problem with that dataset. Both anime characters and selfies are females. And CycleGAN projects transforming male selfies to anime male characters either are not found.

So there are two purposes of this project:
- implement CycleGAN architecture with further training of it on selfie2anime dataset
- trying to collect custom dataset male2anime and train implemented CycleGAN on this dataset for male photo to anime character.


# Training history and detail

### Selfie2anime

CycleGAN was trained without `gradient penalty` and `identity mapping loss` for 190 epoches (75 epoches with 0.0002 lr, 25 with 0.0001 lr, 90 epoch with linear decay lr)

With the following results:



### Male2anime

Custom dataset wad collected. 3050 male anime characters were webscrapped [from anime-planet][anime-planet], 4x upscaled by [waifu2x][https://github.com/yu45020/Waifu2x], faces were detected by [lbpcascade_animeface][https://github.com/nagadomi/lbpcascade_animeface], then resize and centercropped by 256.



[selfie2anime]: https://www.kaggle.com/datasets/arnaud58/selfie2anime
[anime-planet] :https://www.anime-planet.com